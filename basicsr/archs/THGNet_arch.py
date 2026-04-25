import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
import sys
sys.path.insert(0, '/path/to/THGNet')  
from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.TMAE_arch import FeatureMAE_SR
from basicsr.archs.TMAE_arch import FeatureMAE_EncoderOnly
import matplotlib.pyplot as plt
import seaborn as sns


class SCAMAttention(nn.Module):
    """Channel attention + spatial attention."""
    def __init__(self, in_dim, deform_groups=8, kernel_size=3):
        super().__init__()

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//4, in_dim, 1),
            nn.Sigmoid()
        )

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(in_dim)

        # Residual weights
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        # Offset output layer
        # Output channels must follow the BasicVSR++ packing format for (o1, o2, mask): 27 * deform_groups
        self.offset_conv = nn.Conv2d(
            in_dim, 3 * kernel_size * kernel_size * deform_groups, 1
        )

        # Mild initialization
        nn.init.normal_(self.offset_conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # Changed to only accept the concatenated x to avoid mismatch between in_dim and actual channels
        # x: [B, in_dim, H, W]

        # Layer normalization for numerical stability
        b, c, h, w = x.shape
        x_norm = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Channel attention
        ca_weights = self.channel_attention(x_norm)
        x_ca = x_norm * ca_weights

        # Spatial attention
        avg_out = torch.mean(x_norm, dim=1, keepdim=True)
        max_out, _ = torch.max(x_norm, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa_weights = self.spatial_attention(spatial_input)
        x_sa = x_norm * sa_weights

        # Dual-attention fusion
        x_enhanced = self.alpha * x_ca + self.beta * x_sa + x_norm

        # Convert to offsets
        offsets = self.offset_conv(x_enhanced)
        return offsets


class OffsetPredictor(nn.Module):
    """Predict offsets from the reference and supporting frames using global + local prediction, with attention added to the global branch."""
    def __init__(self, n_feats=64, deform_groups=8, kernel_size=3, in_channels=None):
        super(OffsetPredictor, self).__init__()
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups

        # [MOD] in_channels must match the actual input channels:
        #       The BasicVSR++ alignment predictor input is [cond(3C) + flow1(2) + flow2(2)] => 3C + 4
        #       If not provided, it defaults to 3C + 4
        if in_channels is None:
            in_channels = 3 * n_feats + 4
        self.in_channels = in_channels

        # [MOD] Change the global branch input channels to in_channels (no longer fixed to 2 * n_feats)
        self.global_branch = SCAMAttention(self.in_channels, deform_groups, kernel_size)

        # Local branch - uses a dilated convolution pyramid to capture multi-scale local details (unchanged)
        # First layer: regular convolution for dimensionality reduction
        # [MOD] Change the input channels to in_channels (no longer fixed to 2 * n_feats)
        self.local_conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, n_feats // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Three branches of the dilated convolution pyramid
        self.dilation1 = nn.Sequential(
            nn.Conv2d(n_feats // 2, n_feats // 4, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(n_feats // 2, n_feats // 4, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            nn.Conv2d(n_feats // 2, n_feats // 4, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )

        # Fuse multi-scale features
        self.local_fusion = nn.Sequential(
            nn.Conv2d(n_feats // 4 * 3, n_feats // 4, 1),
            nn.ReLU(inplace=True),
            # [MOD] Output channels must be 27 * deform_groups (packed as o1, o2, mask)
            nn.Conv2d(n_feats // 4, 3 * kernel_size * kernel_size * deform_groups, 1)
        )

        # Initialize weights to zero
        nn.init.constant_(self.local_fusion[-1].weight, 0)
        nn.init.constant_(self.local_fusion[-1].bias, 0)

    def forward(self, x):
        # [MOD] Changed to only accept the concatenated x, x: [B, in_channels, H, W]
        #       No longer receives ref_feats/supp_feats to avoid dimension errors caused by changes in external concatenation

        # Global offset prediction - uses the self-attention mechanism
        global_offsets = self.global_branch(x)  # [B, 27*g, H, W]

        # Local offset prediction - uses a dilated convolution pyramid
        x_local = self.local_conv1(x)
        dil1 = self.dilation1(x_local)
        dil2 = self.dilation2(x_local)
        dil3 = self.dilation3(x_local)
        multi_scale = torch.cat([dil1, dil2, dil3], dim=1)
        local_offsets = self.local_fusion(multi_scale)  # [B, 27*g, H, W]

        # Add global and local offsets
        offsets = global_offsets + local_offsets  # [B, 27*g, H, W]

        return offsets


class HighFreqEnhancer(nn.Module):
    """High-frequency enhancement for blur+compression degradation:
       use mild high-pass: hf = x - gaussian_blur(x), then gated fusion.
    """
    def __init__(self, mid_channels=64, ksize=5, sigma=1.0):
        super().__init__()
        self.ksize = ksize

        # fixed gaussian kernel (not trainable)
        grid = torch.arange(ksize).float() - (ksize - 1) / 2
        g = torch.exp(-(grid**2) / (2 * sigma**2))
        g = (g / g.sum()).view(1, 1, -1)  # [1,1,K]
        k2d = (g.transpose(2, 1) @ g).view(1, 1, ksize, ksize)  # [1,1,K,K]
        self.register_buffer("gauss", k2d)

        # encode hf image -> feature space
        self.hf_encoder = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        )

        # gate from current features (suppress noise / flat regions)
        self.gate = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.Sigmoid()
        )

        # stable training: start from 0 and learn enhancement strength
        self.scale = nn.Parameter(torch.tensor(0.0))

    def _blur(self, x):
        k = self.gauss.repeat(3, 1, 1, 1)  # [3,1,K,K]
        pad = self.ksize // 2
        return F.conv2d(x, k, padding=pad, groups=3)

    def forward(self, x_rgb, feat):
        # mild high-pass
        hf = x_rgb - self._blur(x_rgb)          # [B,3,H,W]
        hf_feat = self.hf_encoder(hf)           # [B,C,H,W]
        g = self.gate(feat)                     # [B,C,H,W]
        return feat + self.scale * g * hf_feat



@ARCH_REGISTRY.register()
class THGNet(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped. Besides, we adopt the official DCN
    implementation and the version of torch need to be higher than 1.9.

    ``Paper: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment``

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 MAE_path=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)
    
        # feature extraction module
        if is_low_res_input:
            self.feat_extract = FeatureMAE_EncoderOnly(MAE_path)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.hf_enhance = HighFreqEnhancer(mid_channels)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation \
                (current to previous). 'flows_backward' corresponds to the flows used for backward-time \
                propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                x_i = lqs[:, i, :, :, :]
                feat = self.feat_extract(x_i)
                feat = self.hf_enhance(x_i, feat)     # [ADD] High-frequency enhancement
                feats['spatial'].append(feat.cpu())
                torch.cuda.empty_cache()
        else:
            lqs_ = lqs.view(-1, c, h, w)
            feats_ = self.feat_extract(lqs_)
            feats_ = self.hf_enhance(lqs_, feats_)   # [ADD] Batch enhancement
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        # [MOD] Replace the original conv_offset with your OffsetPredictor, and set the input channels to 3C + 4
        self.conv_offset = OffsetPredictor(
            n_feats=self.out_channels,
            deform_groups=self.deformable_groups,
            kernel_size=3,
            in_channels=3 * self.out_channels + 4
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        # [MOD] conv_offset is no longer the original nn.Sequential, so constant initialization is not applied to conv_offset[-1]
        # _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        # [MOD] The predictor input must be [B, 3C+4, H, W], consistent with the original version
        predictor_in = torch.cat([extra_feat, flow_1, flow_2], dim=1)  # [B, 3C+4, H, W]
        out = self.conv_offset(predictor_in)                           # [B, 27*g, H, W]

        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))

if __name__ == '__main__':
    input = torch.rand(1, 15, 3, 160, 160).cuda()  # B C H W
    model = THGNet().cuda()
    # output = model(input)
    print_network(model)
    flops, params = profile(model, inputs=(input,))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))
    print(torch.cuda.is_available())
    print(list(model.deform_align.keys()))


