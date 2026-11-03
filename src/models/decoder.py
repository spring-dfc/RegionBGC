"""IoU-aware region query selection and decoding with EMHCA."""

import math
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import get_activation, inverse_sigmoid
from .utils import bias_init_with_prob


from src.core import register


__all__ = ['RegionQueryDecoder']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class EnhancedMultiHeadCrossAttention(nn.Module):
    """EMHCA with an additive relative-position bias."""
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        dropout=0.0,
        inside_bias_init: float = 1.5,
        distance_scale_init: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)


        self.inside_bias = nn.Parameter(torch.tensor(float(inside_bias_init), dtype=torch.float32))
        self.distance_scale = nn.Parameter(torch.tensor(float(distance_scale_init), dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.q_proj.weight)
        init.constant_(self.q_proj.bias, 0)
        init.xavier_uniform_(self.k_proj.weight)
        init.constant_(self.k_proj.bias, 0)
        init.xavier_uniform_(self.v_proj.weight)
        init.constant_(self.v_proj.bias, 0)
        init.xavier_uniform_(self.out_proj.weight)
        init.constant_(self.out_proj.bias, 0)

    @staticmethod
    def _build_token_intervals(value_lengths, device, dtype):
        """Normalized (center,length) per memory token: [1, sum(L), 2]."""
        intervals = []
        for w in value_lengths:
            if int(w) < 1:
                raise ValueError('BGC global attention requires positive token counts')
            centers = (torch.arange(int(w), dtype=dtype, device=device) + 0.5) / int(w)
            lengths = torch.full_like(centers, 1.0 / int(w))
            intervals.append(torch.stack((centers, lengths), dim=-1).unsqueeze(0))
        return torch.concat(intervals, dim=1)

    def _build_reference_bias(self, reference_points, token_intervals):
        """Return additive bias with shape [B, 1, Len_q, Len_v]."""
        refs = reference_points
        if refs.ndim != 3 or refs.shape[-1] != 2:
            raise ValueError('BGC references must be (center,length), with last dimension 2')
        tokens = token_intervals.to(device=refs.device, dtype=refs.dtype)
        ref_lengths = refs[..., 1].clamp(min=1e-3).unsqueeze(-1)
        center_delta = tokens[..., 0].unsqueeze(1) - refs[..., 0].unsqueeze(-1)
        inside = center_delta.abs() <= (ref_lengths + tokens[..., 1].unsqueeze(1)) * 0.5
        distance_penalty = (center_delta / ref_lengths).pow(2)
        inside_gain = F.softplus(self.inside_bias)
        distance_scale = F.softplus(self.distance_scale)
        bias = inside.to(refs.dtype) * inside_gain - distance_scale * distance_penalty
        return bias.unsqueeze(1)

    def forward(
        self,
        query,
        reference_points,
        value,
        value_lengths,
        value_mask=None,
        query_pos_embed=None,
        value_pos_embed=None,
    ):
        bs, len_q, _ = query.shape
        len_v = value.shape[1]

        if query_pos_embed is not None:
            query = query + query_pos_embed
        key = value if value_pos_embed is None else value + value_pos_embed

        q = self.q_proj(query).reshape(bs, len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(bs, len_v, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(bs, len_v, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        token_intervals = self._build_token_intervals(value_lengths, value.device, value.dtype)
        attn = attn + self._build_reference_bias(reference_points, token_intervals).to(attn.dtype)

        if value_mask is not None:
            valid_mask = value_mask.to(torch.bool)
            attn = attn.masked_fill(~valid_mask[:, None, None, :], torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(bs, len_q, self.embed_dim)
        output = self.out_proj(output)
        return output


class RegionQueryDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 global_cross_attn_use_reference_bias: bool = True,
                 global_cross_attn_inside_bias_init: float = 1.5,
                 global_cross_attn_distance_scale_init: float = 2.0,):
        super(RegionQueryDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn_type = 'global'
        self.global_cross_attn_use_reference_bias = bool(global_cross_attn_use_reference_bias)
        if self.global_cross_attn_use_reference_bias:
            # Enhanced multi-head cross-attention (EMHCA).
            self.cross_attn = EnhancedMultiHeadCrossAttention(
                d_model,
                n_head,
                dropout=dropout,
                inside_bias_init=global_cross_attn_inside_bias_init,
                distance_scale_init=global_cross_attn_distance_scale_init,
            )
        else:
            # Standard multi-head cross-attention for ablation.
            self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_lengths,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                memory_pos_embed=None):
        """Global interval attention over [B,sum(L),C] sequence memory."""

        q = k = self.with_pos_embed(tgt, query_pos_embed)

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2) Global cross-attention over all memory tokens.
        if self.global_cross_attn_use_reference_bias:
            tgt2 = self.cross_attn(
                tgt,
                reference_points,
                memory,
                memory_lengths,
                memory_mask,
                query_pos_embed=query_pos_embed,
                value_pos_embed=memory_pos_embed,
            )
        else:
            key_padding_mask = None
            if memory_mask is not None:
                key_padding_mask = (~memory_mask.to(torch.bool))
            q = self.with_pos_embed(tgt, query_pos_embed)
            k = self.with_pos_embed(memory, memory_pos_embed)
            tgt2, _ = self.cross_attn(q, k, value=memory, key_padding_mask=key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)


        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)

        tgt = self.norm3(tgt.clamp(min=-65504, max=65504))

        return tgt


class RegionQueryDecoderStack(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx: int = -1):
        super(RegionQueryDecoderStack, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_lengths,
        interval_head,
        score_head,
        query_pos_head,
        memory_pos_embed=None,
        attn_mask=None,
        memory_mask=None,
        type_head=None,
    ):
        """Transformer decoder with optional extra type head.

        When ``type_head`` is provided (a ``nn.ModuleList`` with per-layer heads),
        we additionally return a stacked tensor of type logits per decoder layer.
        This keeps the original interface backward compatible when ``type_head``
        is ``None``.
        """



        output = tgt
        dec_out_intervals = []
        dec_out_logits = []
        dec_out_types = [] if type_head is not None else None

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            # The same [B,Q,2] reference intervals guide all memory scales.
            ref_points_input = ref_points_detach

            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_lengths,
                attn_mask,
                memory_mask,
                query_pos_embed,
                memory_pos_embed,
            )


            inter_ref_interval = F.sigmoid(interval_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if type_head is not None:
                    dec_out_types.append(type_head[i](output))
                if i == 0:
                    dec_out_intervals.append(inter_ref_interval)
                else:

                    dec_out_intervals.append(F.sigmoid(interval_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:

                dec_out_logits.append(score_head[i](output))
                if type_head is not None:
                    dec_out_types.append(type_head[i](output))
                dec_out_intervals.append(inter_ref_interval)
                break


            ref_points = inter_ref_interval

            ref_points_detach = inter_ref_interval.detach() if self.training else inter_ref_interval

        if dec_out_types is not None:
            return (
                torch.stack(dec_out_intervals),
                torch.stack(dec_out_logits),
                torch.stack(dec_out_types),
            )
        return torch.stack(dec_out_intervals), torch.stack(dec_out_logits)


@register
class RegionQueryDecoder(nn.Module):
    __share__ = ['num_classes']

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        eval_sequence_length=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        # Auxiliary supervision for query selection and decoder layers.
        decoder_aux_loss: bool = True,
        encoder_aux_loss: bool = True,
        decoder_aux_weight: float = 1.0,
        encoder_aux_weight: float = 1.0,
        # Product-type auxiliary supervision.
        decoder_aux_bgc_types_loss: bool = True,
        num_bgc_types: int = 0,
        # Disable the relative-position bias for the standard-attention ablation.
        cross_attn_type: str = 'global',
        global_cross_attn_use_reference_bias: bool = True,
        global_cross_attn_inside_bias_init: float = 1.5,
        global_cross_attn_distance_scale_init: float = 2.0,
        use_memory_pos_embed: bool = True,
        memory_pos_temperature: float = 10000.0,
    ):
        super(RegionQueryDecoder, self).__init__()



        assert len(feat_channels) == num_levels
        assert len(feat_strides) == len(feat_channels)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_sequence_length = eval_sequence_length
        self.aux_loss = aux_loss
        self.decoder_aux_loss = bool(decoder_aux_loss)
        self.encoder_aux_loss = bool(encoder_aux_loss)
        self.decoder_aux_weight = float(decoder_aux_weight) if decoder_aux_weight is not None else 1.0
        self.encoder_aux_weight = float(encoder_aux_weight) if encoder_aux_weight is not None else 1.0
        self.decoder_aux_bgc_types_loss = bool(decoder_aux_bgc_types_loss)
        self.num_bgc_types = int(num_bgc_types) if num_bgc_types is not None else 0
        self.cross_attn_type = str(cross_attn_type).lower().strip()
        if self.cross_attn_type != 'global':
            raise ValueError('The BGC interval decoder supports cross_attn_type="global" only')
        self.global_cross_attn_use_reference_bias = bool(global_cross_attn_use_reference_bias)
        self.global_cross_attn_inside_bias_init = float(global_cross_attn_inside_bias_init)
        self.global_cross_attn_distance_scale_init = float(global_cross_attn_distance_scale_init)
        self.use_memory_pos_embed = bool(use_memory_pos_embed)
        self.memory_pos_temperature = float(memory_pos_temperature)

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = RegionQueryDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            global_cross_attn_use_reference_bias=self.global_cross_attn_use_reference_bias,
            global_cross_attn_inside_bias_init=self.global_cross_attn_inside_bias_init,
            global_cross_attn_distance_scale_init=self.global_cross_attn_distance_scale_init,
        )
        self.decoder = RegionQueryDecoderStack(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # decoder embedding
        self.query_pos_head = MLP(2, 2 * hidden_dim, hidden_dim, num_layers=2)

        # Prediction network for IoU-aware region query selection.
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_interval_head = MLP(hidden_dim, hidden_dim, 2, num_layers=3)

        # decoder heads: detection (BGC vs background) + optional multi-label BGC type head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_interval_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 2, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        if self.num_bgc_types > 0:
            self.dec_bgc_type_head = nn.ModuleList([
                nn.Linear(hidden_dim, self.num_bgc_types)
                for _ in range(num_decoder_layers)
            ])
        else:
            self.dec_bgc_type_head = None

        # init encoder output anchors and valid_mask
        if self.eval_sequence_length:
            self.anchors, self.valid_mask = self._generate_anchors()

        # Scale embedding distinguishes sequence resolutions.
        self.level_embed = nn.Parameter(torch.Tensor(self.num_levels, self.hidden_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_interval_head.layers[-1].weight, 0)
        init.constant_(self.enc_interval_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_interval_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        if self.dec_bgc_type_head is not None:
            for head in self.dec_bgc_type_head:
                init.constant_(head.bias, bias)

        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.normal_(self.level_embed)


    @staticmethod
    def build_sincos_position_embedding(w, embed_dim=256, temperature=10000.0):
        # Returns: [1, w, embed_dim]
        assert embed_dim % 2 == 0, 'Embedding dimension must be even for sine/cosine position embeddings'
        grid_w = torch.arange(int(w), dtype=torch.float32)
        pos_dim = embed_dim // 2
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.concat([out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def _build_memory_pos_embed(self, level_lengths, device):
        if not self.use_memory_pos_embed:
            return None
        pos_embeds = []
        for lvl, w in enumerate(level_lengths):
            if int(w) < 1:
                raise ValueError("BGC memory positions require a positive sequence length")
            pos = self.build_sincos_position_embedding(
                w, self.hidden_dim, self.memory_pos_temperature
            ).to(device)
            pos = pos + self.level_embed[lvl].view(1, 1, -1)
            pos_embeds.append(pos)
        return torch.concat(pos_embeds, dim=1)  # [1, sum(L), C]


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv1d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm1d(self.hidden_dim,))])
                )
            )

    def _get_encoder_input(self, feats):
        """Project [B,C,L] features and concatenate sequence memory."""

        if len(feats) != self.num_levels:
            raise ValueError("BGC decoder requires one feature per configured sequence scale")
        if any(feat.ndim != 3 or feat.shape[-1] < 1 for feat in feats):
            raise ValueError("BGC decoder features must have shape [B,C,L], L > 0")
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # Transpose each sequence from [B,C,L] to [B,L,C], then concatenate.
        feat_flatten = []
        level_lengths = []
        for i, feat in enumerate(proj_feats):
            w = feat.shape[-1]
            # [B,C,L] -> [B,L,C]
            feat_flatten.append(feat.transpose(1, 2))
            # One sequence length per scale.
            level_lengths.append(w)

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, level_lengths

    def _generate_anchors(self,
                          level_lengths=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        """Return (center,length) anchor logits [1,sum(L),2] and validity mask.

        Preserve the existing per-level length prior, along the ORF sequence axis.
        Invalid positions keep finite coordinates and are excluded before top-k.
        """
        if level_lengths is None:
            level_lengths = [max(1, int(self.eval_sequence_length) // s) for s in self.feat_strides]
        anchors = []
        for lvl, w in enumerate(level_lengths):
            if int(w) < 1:
                raise ValueError('BGC anchors require positive token counts')
            centers = (torch.arange(int(w), dtype=dtype, device=device) + 0.5) / int(w)
            lengths = torch.full_like(centers, grid_size * (2.0 ** lvl))
            anchors.append(torch.stack((centers, lengths), dim=-1).unsqueeze(0))
        anchors = torch.concat(anchors, dim=1)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(-1, keepdim=True)
        return inverse_sigmoid(anchors), valid_mask


    def _select_region_queries(self,
                           memory,
                           level_lengths):
        """Select top-k encoder tokens and initialize normalized interval queries."""
        bs, _, _ = memory.shape
        # prepare input for decoder
        expected_lengths = [max(1, int(self.eval_sequence_length) // s) for s in self.feat_strides] if self.eval_sequence_length else None
        if self.training or list(level_lengths) != expected_lengths:
            anchors, valid_mask = self._generate_anchors(level_lengths, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)


        memory = valid_mask.to(memory.dtype) * memory


        output_memory = self.enc_output(memory)


        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_interval_head(output_memory) + anchors

        # Exclude invalid anchors even after biased encoder projections.
        valid_count = int(valid_mask.sum().item())
        k = min(self.num_queries, valid_count)
        if k == 0:
            raise ValueError('No valid BGC interval anchors for the input sequence')

        selection_scores = enc_outputs_class.max(-1).values.masked_fill(~valid_mask.squeeze(-1), -torch.inf)
        _, topk_ind = torch.topk(selection_scores, k, dim=1)


        reference_points_unact = enc_outputs_coord_unact.gather(dim=1,\
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        # Supervise exactly the candidates selected for decoder initialization.
        enc_aux_intervals = reference_points_unact.sigmoid()
        enc_aux_logits = enc_outputs_class.gather(dim=1,\
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # Initialize queries from detached selected encoder features.
        target = output_memory.gather(dim=1,\
            index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        target = target.detach()

        return target, reference_points_unact.detach(), enc_aux_intervals, enc_aux_logits


    def forward(self, feats, targets=None):

        """Process ORF-axis features using the tensor layout documented by this module."""

        memory, level_lengths = self._get_encoder_input(feats)
        memory_pos_embed = self._build_memory_pos_embed(level_lengths, device=memory.device)


        target, init_ref_points_unact, enc_aux_intervals, enc_aux_logits =\
            self._select_region_queries(memory, level_lengths)


        decoder_out = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            level_lengths,
            self.dec_interval_head,
            self.dec_score_head,
            self.query_pos_head,
            memory_pos_embed=memory_pos_embed,
            type_head=self.dec_bgc_type_head,
        )
        if self.dec_bgc_type_head_is_enabled:
            out_intervals, out_logits, out_bgc_types = decoder_out
        else:
            out_intervals, out_logits = decoder_out
            out_bgc_types = None


        out = {
            'pred_logits': out_logits[-1],
            'pred_intervals': out_intervals[-1],
        }
        if out_bgc_types is not None:
            out['pred_bgc_types'] = out_bgc_types[-1]


        if self.training and self.aux_loss:
            # (A) decoder intermediate-layer auxiliary outputs
            if self.decoder_aux_loss and self.decoder_aux_weight != 0.0:
                # Include multi-label head logits for intermediate layers when enabled,
                # so RegionBGCCriterion can compute loss_bgc_types on aux outputs.
                if out_bgc_types is not None and self.decoder_aux_bgc_types_loss:
                    out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_intervals[:-1], out_bgc_types[:-1])
                else:
                    out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_intervals[:-1])
                out['aux_weight'] = self.decoder_aux_weight
            # (B) encoder candidate auxiliary outputs (enc_score_head / enc_interval_head)
            if self.encoder_aux_loss and self.encoder_aux_weight != 0.0:
                out['enc_aux_outputs'] = self._set_aux_loss([enc_aux_logits], [enc_aux_intervals])
                out['enc_aux_weight'] = self.encoder_aux_weight

        return out


    @property
    def dec_bgc_type_head_is_enabled(self) -> bool:

        return getattr(self, 'dec_bgc_type_head', None) is not None

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_bgc_types=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_bgc_types is None:
            return [
                {'pred_logits': a, 'pred_intervals': b}
                for a, b in zip(outputs_class, outputs_coord)
            ]
        return [
            {'pred_logits': a, 'pred_intervals': b, 'pred_bgc_types': t}
            for a, b, t in zip(outputs_class, outputs_coord, outputs_bgc_types)
        ]
