"""Multi-scale region feature encoder: MST, ISFI and CSFF."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from src.core import register


__all__ = ['MultiScaleRegionFeatureEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()

        # Kernel, stride and padding refer only to the ORF sequence axis.
        self.conv = nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm1d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        # x: [B,C_in,L] -> [B,C_out,L_out]
        return self.act(self.norm(self.conv(x)))


class RepVGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        # Two training branches: width-3 and pointwise sequence convolutions.
        # Branch 1: kernel size 3
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        # Branch 2: kernel size 1
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):

        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            # Fuse both branches into one width-3 sequence convolution.
            self.conv = nn.Conv1d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3, bias3 = self._fuse_bn_tensor(self.conv1)
        kernel1, bias1 = self._fuse_bn_tensor(self.conv2)

        return kernel3 + self._pad_pointwise_kernel(kernel1), bias3 + bias1

    def _pad_pointwise_kernel(self, kernel1):
        if kernel1 is None:
            return 0
        else:
            # Pad the pointwise kernel to width 3 before summing branches.
            return F.pad(kernel1, [1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CrossScaleFusionBlock(nn.Module):
    """CSFF fusion block."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CrossScaleFusionBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):

        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class IntraScaleFeatureInteractionLayer(nn.Module):
    """ISFI self-attention and feed-forward layer."""
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        # src: [B, L, C]

        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class IntraScaleFeatureInteraction(nn.Module):
    """Intra-Scale Feature Interaction (ISFI)."""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(IntraScaleFeatureInteraction, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:

        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class MultiScaleRegionFeatureEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_sequence_length=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_sequence_length = eval_sequence_length

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides


        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(hidden_dim)
                )
            )

        # ISFI on the coarsest retained scale.

        encoder_layer = IntraScaleFeatureInteractionLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            IntraScaleFeatureInteraction(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])


        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CrossScaleFusionBlock(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )


        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            # Downsample the sequence with kernel=3, stride=2, padding=1.
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, padding=1, act=act)
            )
            self.pan_blocks.append(
                CrossScaleFusionBlock(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_sequence_length:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                w = max(1, int(self.eval_sequence_length) // stride)
                pos_embed = self.build_sincos_position_embedding(
                    w, self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)


    @staticmethod
    def build_sincos_position_embedding(w, embed_dim=256, temperature=10000.):
        """Build sine/cosine embeddings for positions on the ORF sequence axis."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        assert embed_dim % 2 == 0, 'Embedding dimension must be even for sine/cosine position embeddings'
        pos_dim = embed_dim // 2
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        """Encode and fuse sequence features [B,C,L] at each retained scale."""
        assert len(feats) == len(self.in_channels)
        if any(feat.ndim != 3 or feat.shape[-1] < 1 for feat in feats):
            raise ValueError("BGC encoder features must have shape [B,C,L], L > 0")

        proj_feats = self._multi_scale_transformation(feats)
        proj_feats = self._intra_scale_feature_interaction(proj_feats)
        return self._cross_scale_feature_fusion(proj_feats)

    def _multi_scale_transformation(self, feats):
        """Complete MST by projecting the pooled gene embeddings."""
        return [self.input_proj[i](feat) for i, feat in enumerate(feats)]

    def _intra_scale_feature_interaction(self, proj_feats):
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                # Sequence feature: [B,hidden_dim,L].
                w = proj_feats[enc_ind].shape[-1]
                # Attention layout: [B,L,C].
                src_flatten = proj_feats[enc_ind].transpose(1, 2)
                cached_pos = getattr(self, f'pos_embed{enc_ind}', None)
                if self.training or cached_pos is None or cached_pos.shape[1] != w:
                    pos_embed = self.build_sincos_position_embedding(
                        w, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:

                    pos_embed = cached_pos.to(src_flatten.device)

                # memory: [B, L, C]
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                # Return to convolution layout [B,C,L].
                proj_feats[enc_ind] = memory.transpose(1, 2).contiguous()

        return proj_feats

    def _cross_scale_feature_fusion(self, proj_feats):
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            # Align sequence lengths explicitly, including odd input lengths.
            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[-1], mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            # Pooling and strided convolutions round odd lengths differently.
            if downsample_feat.shape[-1] != feat_high.shape[-1]:
                downsample_feat = F.interpolate(downsample_feat, size=feat_high.shape[-1], mode='nearest')
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)
        return outs
