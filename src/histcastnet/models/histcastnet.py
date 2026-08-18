# Derived from Earthformer (Apache License 2.0).
# Modified for HistCastNet.
from collections import OrderedDict
import torch
from torch import nn
from torch.cuda.amp import autocast

from .utils import (
    get_activation,
    get_norm_layer,
    _generalize_padding,
    _generalize_unpadding,
    apply_initialization,
)
from ..layers.DWT_IDWT_layer import FrameWiseDWT2D, FrameWiseIDWT2D


# -------------------------------------------------


# -------------------------------------------------
class PositionwiseFFN(nn.Module):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    Also, if we use gated projection. We will use
        fc1_1 * act(fc1_2(data)) to map the data
    """

    def __init__(
        self,
        units: int = 512,
        hidden_size: int = 2048,
        activation_dropout: float = 0.0,
        dropout: float = 0.1,
        gated_proj: bool = False,
        activation="relu",
        normalization: str = "layer_norm",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super().__init__()
        # initialization
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        self._pre_norm = pre_norm
        self._gated_proj = gated_proj
        self._kwargs = OrderedDict(
            [
                ("units", units),
                ("hidden_size", hidden_size),
                ("activation_dropout", activation_dropout),
                ("activation", activation),
                ("dropout", dropout),
                ("normalization", normalization),
                ("layer_norm_eps", layer_norm_eps),
                ("gated_proj", gated_proj),
                ("pre_norm", pre_norm),
            ]
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_dropout_layer = nn.Dropout(activation_dropout)
        self.ffn_1 = nn.Linear(in_features=units, out_features=hidden_size, bias=True)
        if self._gated_proj:
            self.ffn_1_gate = nn.Linear(
                in_features=units, out_features=hidden_size, bias=True
            )
        self.activation = get_activation(activation)
        self.ffn_2 = nn.Linear(in_features=hidden_size, out_features=units, bias=True)
        self.layer_norm = get_norm_layer(
            normalization=normalization, in_channels=units, epsilon=layer_norm_eps
        )
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.ffn_1, linear_mode=self.linear_init_mode)
        if self._gated_proj:
            apply_initialization(self.ffn_1_gate, linear_mode=self.linear_init_mode)
        apply_initialization(self.ffn_2, linear_mode=self.linear_init_mode)
        apply_initialization(self.layer_norm, norm_mode=self.norm_init_mode)

    def forward(self, data):
        """
        data: (B, seq_length, C_in)
        return: (B, seq_length, C_out)
        """

        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out


# -------------------------------------------------
# PatchMerging / Upsample / DownSampling
# -------------------------------------------------
class PatchMerging3D(nn.Module):
    """Patch Merging Layer"""

    def __init__(
        self,
        dim,
        out_dim=None,
        downsample=(1, 2, 2),
        norm_layer="layer_norm",
        padding_type="nearest",
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        if out_dim is None:
            out_dim = max(downsample) * dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.padding_type = padding_type
        self.reduction = nn.Linear(
            downsample[0] * downsample[1] * downsample[2] * dim, out_dim, bias=False
        )
        self.norm = get_norm_layer(
            norm_layer, in_channels=downsample[0] * downsample[1] * downsample[2] * dim
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(
                m, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode
            )

    def get_out_shape(self, data_shape):
        T, H, W, C_in = data_shape
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        return (
            (T + pad_t) // self.downsample[0],
            (H + pad_h) // self.downsample[1],
            (W + pad_w) // self.downsample[2],
            self.out_dim,
        )

    def forward(self, x):
        """
        x: (B, T, H, W, C)
        return: (B, T//ds[0], H//ds[1], W//ds[2], out_dim)
        """
        B, T, H, W, C = x.shape

        # padding
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        if pad_h or pad_t or pad_w:
            T += pad_t
            H += pad_h
            W += pad_w
            x = _generalize_padding(
                x, pad_t, pad_h, pad_w, padding_type=self.padding_type
            )

        x = (
            x.reshape(
                (
                    B,
                    T // self.downsample[0],
                    self.downsample[0],
                    H // self.downsample[1],
                    self.downsample[1],
                    W // self.downsample[2],
                    self.downsample[2],
                    C,
                )
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(
                B,
                T // self.downsample[0],
                H // self.downsample[1],
                W // self.downsample[2],
                self.downsample[0] * self.downsample[1] * self.downsample[2] * C,
            )
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Upsample3DLayer(nn.Module):
    """Upsampling based on nn.Upsample and Conv2d."""

    def __init__(
        self,
        dim,
        out_dim,
        target_size,
        temporal_upsample=False,
        kernel_size=3,
        layout="THWC",
        conv_init_mode="0",
    ):
        super(Upsample3DLayer, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size  # (T_new, H_new, W_new)
        self.out_dim = out_dim
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up = nn.Upsample(size=target_size, mode="nearest")  # 3D upsampling
        else:
            self.up = nn.Upsample(
                size=(target_size[1], target_size[2]), mode="nearest"
            )  # 2D upsampling
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2),
        )
        assert layout in ["THWC", "CTHW"]
        self.layout = layout

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, conv_mode=self.conv_init_mode)

    def forward(self, x):
        """
        x: (B, T, H, W, C) or (B, C, T, H, W)
        return: same layout, H/W upsampled to target_size
        """

        if self.layout == "THWC":
            B, T, H, W, C = x.shape
            if self.temporal_upsample:
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
                return self.conv(self.up(x)).permute(0, 2, 3, 4, 1)
            else:
                assert self.target_size[0] == T
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, C, H, W)
                x = self.up(x)
                return (
                    self.conv(x)
                    .permute(0, 2, 3, 1)
                    .reshape((B,) + self.target_size + (self.out_dim,))
                )
        elif self.layout == "CTHW":
            B, C, T, H, W = x.shape
            if self.temporal_upsample:
                return self.conv(self.up(x))
            else:
                assert self.target_size[0] == T
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                x = x.reshape(B * T, C, H, W)
                return (
                    self.conv(self.up(x))
                    .reshape(
                        B,
                        self.target_size[0],
                        self.out_dim,
                        self.target_size[1],
                        self.target_size[2],
                    )
                    .permute(0, 2, 1, 3, 4)
                )


# -------------------------------------------------
# Positional Embeddings
# -------------------------------------------------
class PosEmbed(nn.Module):

    def __init__(self, embed_dim, maxT, maxH, maxW, typ="t+h+w"):
        r"""
        typ:
            - t+h+w
            - t+hw
        """
        super(PosEmbed, self).__init__()
        self.typ = typ

        assert self.typ in ["t+h+w", "t+hw"]
        self.maxT = maxT
        self.maxH = maxH
        self.maxW = maxW
        self.embed_dim = embed_dim
        # spatiotemporal learned positional embedding
        if self.typ == "t+h+w":
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.H_embed = nn.Embedding(num_embeddings=maxH, embedding_dim=embed_dim)
            self.W_embed = nn.Embedding(num_embeddings=maxW, embedding_dim=embed_dim)
        elif self.typ == "t+hw":
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.HW_embed = nn.Embedding(
                num_embeddings=maxH * maxW, embedding_dim=embed_dim
            )
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, embed_mode="0")

    def forward(self, x):
        """
        x: (B, T, H, W, C)
        return: x + pos
        """
        _, T, H, W, _ = x.shape
        t_idx = torch.arange(T, device=x.device)
        h_idx = torch.arange(H, device=x.device)
        w_idx = torch.arange(W, device=x.device)
        if self.typ == "t+h+w":
            return (
                x
                + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim)
                + self.H_embed(h_idx).reshape(1, H, 1, self.embed_dim)
                + self.W_embed(w_idx).reshape(1, 1, W, self.embed_dim)
            )
        elif self.typ == "t+hw":
            spatial_idx = h_idx.unsqueeze(-1) * self.maxW + w_idx
            return (
                x
                + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim)
                + self.HW_embed(spatial_idx)
            )
        else:
            raise NotImplementedError


class SpacePosEmbed(nn.Module):

    def __init__(self, embed_dim, maxH, maxW):
        super().__init__()
        self.embed_dim = embed_dim
        self.maxH = maxH
        self.maxW = maxW

        self.H_embed = nn.Embedding(maxH, embed_dim)
        self.W_embed = nn.Embedding(maxW, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.H_embed, embed_mode="0")
        apply_initialization(self.W_embed, embed_mode="0")

    def forward(self, x):
        """
        x: (B, T, H, W, C)
        """

        B, T, H, W, C = x.shape
        assert C == self.embed_dim, f"SpacePosEmbed: C={C}, embed_dim={self.embed_dim}"
        assert (
            H <= self.maxH and W <= self.maxW
        ), f"H={H}, W={W} exceeds maxH={self.maxH}, maxW={self.maxW}."

        device = x.device
        h_idx = torch.arange(H, device=device)
        w_idx = torch.arange(W, device=device)

        H_pos = self.H_embed(h_idx).reshape(1, 1, H, 1, C)  # (1,1,H,1,C)
        W_pos = self.W_embed(w_idx).reshape(1, 1, 1, W, C)  # (1,1,1,W,C)

        pos = H_pos + W_pos
        return x + pos


class TimePosEmbed(nn.Module):

    def __init__(self, embed_dim, maxT):
        super(TimePosEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.maxT = maxT
        self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, embed_mode="0")

    def forward(self, T, device):

        assert T <= self.maxT, f"T={T} > maxT={self.maxT}"
        t_idx = torch.arange(T, device=device)
        pos = self.T_embed(t_idx)  # (T, C)
        return pos.unsqueeze(0)  # (1, T, C)


# -------------------------------------------------
# Temporal / Spatial Attention Blocks
# -------------------------------------------------
class TemporalSelfAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, max_len=32, attn_drop=0.0, ffn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,  # (batch, seq, dim)
        )

        self.time_pos_embed = TimePosEmbed(embed_dim=dim, maxT=max_len)

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = PositionwiseFFN(
            units=dim,
            hidden_size=4 * dim,
            activation_dropout=0.0,
            dropout=ffn_drop,
            gated_proj=False,
            activation="gelu",
            normalization="layer_norm",
            pre_norm=True,
            linear_init_mode="0",
            norm_init_mode="0",
        )

    def forward(self, x):

        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        assert (
            T <= self.max_len
        ), f"Temporal length T={T} exceeds max_len={self.max_len} for TimePosEmbed!"

        # (B,H,W,T,C) -> (B*H*W, T, C)
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)

        pos = self.time_pos_embed(T, x_flat.device)  # (1, T, C)
        x_flat = x_flat + pos

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # (B*H*W, T, C)
        x_flat = x_flat + attn_out

        x_flat = self.ffn(x_flat)

        x_out = x_flat.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)  # (B,T,H,W,C)
        return x_out


class WindowSpatialPosEmbed(nn.Module):

    def __init__(self, dim, window_size):
        super().__init__()
        Wh, Ww = window_size
        self.Wh = Wh
        self.Ww = Ww
        self.H_embed = nn.Embedding(Wh, dim)
        self.W_embed = nn.Embedding(Ww, dim)

    def forward(self, x_win):
        N_win, M, C = x_win.shape
        assert (
            M == self.Wh * self.Ww
        ), f"Window token count M={M} does not match Wh*Ww={self.Wh * self.Ww}."

        device = x_win.device
        h_idx = torch.arange(self.Wh, device=device)
        w_idx = torch.arange(self.Ww, device=device)

        pos = self.H_embed(h_idx).unsqueeze(1) + self.W_embed(w_idx).unsqueeze(
            0
        )  # (Wh,Ww,C)
        pos = pos.reshape(1, M, C)  # (1,M,C)
        return x_win + pos


class WindowSpatialAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=(4, 4), attn_drop=0.0, ffn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size  # (Wh, Ww)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,  # (N,M,C)
        )

        self.pos_embed = WindowSpatialPosEmbed(dim, window_size)

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = PositionwiseFFN(
            units=dim,
            hidden_size=4 * dim,
            activation_dropout=0.0,
            dropout=ffn_drop,
            gated_proj=False,
            activation="gelu",
            normalization="layer_norm",
            pre_norm=True,
            linear_init_mode="0",
            norm_init_mode="0",
        )

    def forward(self, x):

        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        Wh, Ww = self.window_size
        assert (
            H % Wh == 0 and W % Ww == 0
        ), f"H, W={H, W} must be divisible by window_size={self.window_size}."

        # (B*T, H, W, C)
        x_bt = x.reshape(B * T, H, W, C)
        nWh, nWw = H // Wh, W // Ww

        # (B*T, nWh, Wh, nWw, Ww, C) -> (B*T, nWh, nWw, Wh, Ww, C)

        x_windows = x_bt.reshape(B * T, nWh, Wh, nWw, Ww, C).permute(0, 1, 3, 2, 4, 5)

        x_windows = x_windows.reshape(B * T * nWh * nWw, Wh * Ww, C)

        x_windows = self.pos_embed(x_windows)

        x_norm = self.norm1(x_windows)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_windows = x_windows + attn_out

        x_windows = self.ffn(x_windows)

        x_windows = x_windows.reshape(B * T, nWh, nWw, Wh, Ww, C).permute(
            0, 1, 3, 2, 4, 5
        )
        x_bt = x_windows.reshape(B * T, H, W, C)

        x_out = x_bt.reshape(B, T, H, W, C)
        return x_out


class DualAttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        max_len=32,
        window_size=(4, 4),
        attn_drop=0.0,
        ffn_drop=0.0,
        use_temporal: bool = True,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.temporal_block = TemporalSelfAttentionBlock(
            dim=dim,
            max_len=max_len,
            num_heads=num_heads,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
        )
        self.spatial_block = WindowSpatialAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
        )

    def forward(self, x):

        if self.use_temporal:
            x = self.temporal_block(x)
        if self.use_spatial:
            x = self.spatial_block(x)
        return x

    def reset_parameters(self):
        if hasattr(self.temporal_block, "reset_parameters"):
            self.temporal_block.reset_parameters()
        if hasattr(self.spatial_block, "reset_parameters"):
            self.spatial_block.reset_parameters()


# -------------------------------------------------
# DualAtt Encoder
# -------------------------------------------------
class DualAttEncoder(nn.Module):

    def __init__(
        self,
        in_channels=4,
        base_dim=64,
        stage1_dim=128,
        num_blocks_stage0=2,
        num_blocks_stage1=2,
        num_heads=4,
        window_size=(4, 4),
        max_T=25,
        max_H=64,
        max_W=64,
        attn_drop=0.0,
        ffn_drop=0.0,
        use_temporal_attn: bool = True,
        use_spatial_attn: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_dim = base_dim
        self.stage1_dim = stage1_dim
        self.max_T = max_T
        self.max_H = max_H
        self.max_W = max_W
        self.use_temporal_attn = use_temporal_attn
        self.use_spatial_attn = use_spatial_attn

        self.proj_in = nn.Linear(in_channels, base_dim)

        self.pos_s0 = SpacePosEmbed(
            embed_dim=base_dim,
            maxH=max_H,
            maxW=max_W,
        )

        self.pos_s1 = SpacePosEmbed(
            embed_dim=stage1_dim,
            maxH=max_H // 2,
            maxW=max_W // 2,
        )

        # Stage 0 blocks
        self.stage0_blocks = nn.ModuleList(
            [
                DualAttentionBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    attn_drop=attn_drop,
                    ffn_drop=ffn_drop,
                    use_temporal=use_temporal_attn,
                    use_spatial=use_spatial_attn,
                )
                for _ in range(num_blocks_stage0)
            ]
        )

        # Downsample to stage1
        self.down = PatchMerging3D(
            dim=base_dim,
            out_dim=stage1_dim,
            downsample=(1, 2, 2),
            padding_type="nearest",
        )

        # Stage 1 blocks
        self.stage1_blocks = nn.ModuleList(
            [
                DualAttentionBlock(
                    dim=stage1_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    attn_drop=attn_drop,
                    ffn_drop=ffn_drop,
                    use_temporal=use_temporal_attn,
                    use_spatial=use_spatial_attn,
                )
                for _ in range(num_blocks_stage1)
            ]
        )

    def forward(self, x):
        """
        x: (B, T_in, H_in, W_in, C_in)
        return:
            feat_s1: (B, T_in, H_in/2, W_in/2, stage1_dim)
            feat_s0: (B, T_in, H_in,   W_in,   base_dim)
        """
        B, T_in, H, W, C = x.shape
        assert (
            C == self.in_channels
        ), f"Input channels C={C} do not match in_channels={self.in_channels}."

        assert T_in <= self.max_T, f"T_in={T_in} > max_T={self.max_T}"
        assert H <= self.max_H, f"H={H} > max_H={self.max_H}"
        assert W <= self.max_W, f"W={W} > max_W={self.max_W}"

        x = self.proj_in(x)  # (B, T_in, H, W, base_dim)

        x = self.pos_s0(x)
        for blk in self.stage0_blocks:
            x = blk(x)
        feat_s0 = x  # (B, T_in, H, W, base_dim)

        x_down = self.down(feat_s0)  # (B, T_in, H/2, W/2, stage1_dim)

        _, _, H2, W2, _ = x_down.shape
        assert H2 <= self.max_H // 2 and W2 <= self.max_W // 2

        x_down = self.pos_s1(x_down)
        for blk in self.stage1_blocks:
            x_down = blk(x_down)
        feat_s1 = x_down  # (B, T_in, H/2, W/2, stage1_dim)

        return feat_s1, feat_s0


# -------------------------------------------------


# -------------------------------------------------
class TemporalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0):
        super().__init__()

        self.spatial_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ffn = PositionwiseFFN(
            units=dim,
            hidden_size=4 * dim,
            activation_dropout=0.0,
            dropout=0.0,
            gated_proj=False,
            activation="gelu",
            normalization="layer_norm",
            pre_norm=True,
            linear_init_mode="0",
            norm_init_mode="0",
        )

    def forward(self, q, kv):
        B, T_q, H, W, C = q.shape
        B, T_kv, _, _, _ = kv.shape

        kv_spatial = kv.reshape(B * T_kv, H, W, C).permute(0, 3, 1, 2).contiguous()

        kv_spatial = self.spatial_conv(kv_spatial)

        kv = kv_spatial.permute(0, 2, 3, 1).reshape(B, T_kv, H, W, C)
        # -------------------------------

        q_flat = q.permute(0, 2, 3, 1, 4).reshape(B * H * W, T_q, C)
        kv_flat = kv.permute(0, 2, 3, 1, 4).reshape(B * H * W, T_kv, C)

        q_norm = self.norm_q(q_flat)
        kv_norm = self.norm_kv(kv_flat)

        # Cross Attention
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        x = q_flat + attn_out

        x = self.ffn(x)

        out = x.reshape(B, H, W, T_q, C).permute(0, 3, 1, 2, 4)
        return out


# -------------------------------------------------


# -------------------------------------------------
class DualAttUNetDecoder(nn.Module):
    def __init__(
        self,
        enc_dim_stage1,  # stage1_dim
        enc_dim_stage0,  # base_dim
        T_out,
        hw_enc,
        hw_out,
        out_channels_pix=1,
        num_heads=4,
        window_size=(4, 4),
        use_temporal_attn: bool = True,
        use_spatial_attn: bool = True,
        fshrd_variant: str = "full",
    ):
        super().__init__()

        valid_fshrd_variants = {
            "full",
            "state_independent",
            "last_memory",
            "high_full_memory",
        }
        if fshrd_variant not in valid_fshrd_variants:
            raise ValueError(
                f"Unsupported fshrd_variant={fshrd_variant!r}; "
                f"expected one of {sorted(valid_fshrd_variants)}."
            )

        self.T_out = T_out
        self.H_enc, self.W_enc = hw_enc
        self.H_out, self.W_out = hw_out
        self.out_channels_pix = out_channels_pix
        self.C_wave = 4 * out_channels_pix
        self.fshrd_variant = fshrd_variant

        self.s1_cross_attn = TemporalCrossAttention(
            dim=enc_dim_stage1, num_heads=num_heads
        )
        self.s0_cross_attn = TemporalCrossAttention(
            dim=enc_dim_stage0, num_heads=num_heads
        )

        # Stage1 decoder blocks
        self.dec_s1_blocks = nn.ModuleList(
            [
                DualAttentionBlock(
                    dim=enc_dim_stage1,
                    num_heads=num_heads,
                    window_size=window_size,
                    use_temporal=use_temporal_attn,
                    use_spatial=use_spatial_attn,
                )
                for _ in range(2)
            ]
        )

        self.up_s1_to_s0 = Upsample3DLayer(
            dim=enc_dim_stage1,
            out_dim=enc_dim_stage0,
            target_size=(1, self.H_out, self.W_out),
            temporal_upsample=False,
            layout="THWC",
        )

        self.dec_s0_blocks = nn.ModuleList(
            [
                DualAttentionBlock(
                    dim=enc_dim_stage0 * 2,
                    num_heads=num_heads,
                    window_size=window_size,
                    use_temporal=use_temporal_attn,
                    use_spatial=use_spatial_attn,
                )
                for _ in range(2)
            ]
        )

        self.head_LL = nn.Linear(enc_dim_stage0 * 2, out_channels_pix)
        self.head_LH = nn.Linear(enc_dim_stage0 * 2, out_channels_pix)
        self.head_HL = nn.Linear(enc_dim_stage0 * 2, out_channels_pix)
        self.head_HH = nn.Linear(enc_dim_stage0 * 2, out_channels_pix)

        nn.init.zeros_(self.head_LH.weight)
        nn.init.zeros_(self.head_LH.bias)
        nn.init.zeros_(self.head_HL.weight)
        nn.init.zeros_(self.head_HL.bias)
        nn.init.zeros_(self.head_HH.weight)
        nn.init.zeros_(self.head_HH.bias)
        # ------------------------------------

        self.future_t_embed = nn.Embedding(T_out, enc_dim_stage1)

    def forward(self, feat_s1, feat_s0):
        B, T_in, H1, W1, C1 = feat_s1.shape
        _, _, H0, W0, C0 = feat_s0.shape

        h_base = feat_s1[:, -1].unsqueeze(1)  # (B,1,H1,W1,C1)
        outputs = []

        # FSHRD ablations only alter the recurrent query and/or the explicit
        # retrieval-memory length. All cross-attention parameters and decoder
        # blocks remain present, so every variant has the same state_dict.
        if self.fshrd_variant == "last_memory":
            low_memory = feat_s1[:, -1:]
        else:
            low_memory = feat_s1

        if self.fshrd_variant == "high_full_memory":
            high_memory = feat_s0
        else:
            high_memory = feat_s0[:, -1:]

        for t in range(self.T_out):

            t_query = self.future_t_embed(
                torch.tensor(t, device=h_base.device)
            ).reshape(1, 1, 1, 1, -1)
            if self.fshrd_variant == "state_independent":
                q_t = t_query.expand(B, 1, H1, W1, C1)
            else:
                q_t = h_base + t_query  # (B,1,H1,W1,C1)

            h_t = self.s1_cross_attn(q_t, low_memory)

            for blk in self.dec_s1_blocks:
                h_t = blk(h_t)

            h_up = self.up_s1_to_s0(h_t)  # (B,1,H0,W0,C0)

            # 5) Stage0 refinement retrieves the last contextualized high-resolution token.
            #    The high_full_memory ablation instead exposes all observed tokens.
            dynamic_skip = self.s0_cross_attn(h_up, high_memory)

            dec_s0 = torch.cat([h_up, dynamic_skip], dim=-1)

            for blk in self.dec_s0_blocks:
                dec_s0 = blk(dec_s0)

            B2, _, H2, W2, C2 = dec_s0.shape
            f = dec_s0.reshape(B2 * H2 * W2, C2)

            LL, LH = self.head_LL(f), self.head_LH(f)
            HL, HH = self.head_HL(f), self.head_HH(f)

            wavelet_frame = torch.stack([LL, LH, HL, HH], dim=-1)
            wavelet_frame = wavelet_frame.reshape(
                B, 1, H2, W2, 4 * self.out_channels_pix
            )

            outputs.append(wavelet_frame)

            h_base = h_t

        return torch.cat(outputs, dim=1)


# -------------------------------------------------
# Wavelet DualAtt Model (encoder + B1 decoder)
# -------------------------------------------------
class HistCastNetBackbone(nn.Module):

    def __init__(
        self,
        input_shape,
        target_shape,
        base_dim=64,
        stage1_dim=128,
        num_blocks_stage0=2,
        num_blocks_stage1=2,
        num_heads=4,
        window_size=(4, 4),
        attn_drop=0.0,
        ffn_drop=0.0,
        use_temporal_attn: bool = True,
        use_spatial_attn: bool = True,
        fshrd_variant: str = "full",
    ):
        super().__init__()

        T_in, H_dwt, W_dwt, C_in = input_shape
        T_out, H2, W2, C_out = target_shape
        assert C_in == C_out, f"C_in={C_in} and C_out={C_out} must match."
        assert (
            H_dwt == H2 and W_dwt == W2
        ), "Input and output wavelet resolutions must match."
        assert (
            H_dwt % 2 == 0 and W_dwt % 2 == 0
        ), "H_dwt and W_dwt must be divisible by 2."

        self.input_shape = input_shape
        self.target_shape = target_shape
        self.C_pix = C_in // 4

        # Encoder
        self.encoder = DualAttEncoder(
            in_channels=C_in,
            base_dim=base_dim,
            stage1_dim=stage1_dim,
            num_blocks_stage0=num_blocks_stage0,
            num_blocks_stage1=num_blocks_stage1,
            num_heads=num_heads,
            window_size=window_size,
            max_T=T_in,
            max_H=H_dwt,
            max_W=W_dwt,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
            use_temporal_attn=use_temporal_attn,
            use_spatial_attn=use_spatial_attn,
        )

        self.decoder = DualAttUNetDecoder(
            enc_dim_stage1=stage1_dim,
            enc_dim_stage0=base_dim,
            T_out=T_out,
            hw_enc=(H_dwt // 2, W_dwt // 2),
            hw_out=(H_dwt, W_dwt),
            out_channels_pix=self.C_pix,
            num_heads=num_heads,
            window_size=window_size,
            use_temporal_attn=use_temporal_attn,
            use_spatial_attn=use_spatial_attn,
            fshrd_variant=fshrd_variant,
        )

    def forward(self, x):
        """
        x: (B, T_in, H_dwt, W_dwt, 4*C_pix)
        """
        feat_s1, feat_s0 = self.encoder(x)
        y_wavelet = self.decoder(feat_s1=feat_s1, feat_s0=feat_s0)
        return y_wavelet


# -------------------------------------------------


# -------------------------------------------------
class HistCastNet(nn.Module):

    def __init__(self, inner_model, dwt_layer, idwt_layer):
        """
        inner_model : HistCastNetBackbone
        dwt_layer   : FrameWiseDWT2D
        idwt_layer  : FrameWiseIDWT2D
        """
        super().__init__()
        self.inner_model = inner_model
        self.dwt = dwt_layer
        self.idwt = idwt_layer

    def forward(self, x):
        # x: (B,T_in,H,W,C_pix)
        #

        #

        with autocast(enabled=False):
            x_dwt_f32 = self.dwt(x.float())  # (B,T_in,H/2,W/2,4*C_pix), float32

        x_dwt = x_dwt_f32.to(dtype=x.dtype)
        y_dwt = self.inner_model(x_dwt)

        with autocast(enabled=False):
            y_f32 = self.idwt(y_dwt.float())  # (B,T_out,H,W,C_pix), float32

        return y_f32.to(dtype=y_dwt.dtype)


# -------------------------------------------------


# -------------------------------------------------
class PixelDualAttDecoder(nn.Module):
    def __init__(
        self,
        enc_dim_stage1,  # stage1_dim
        enc_dim_stage0,  # base_dim
        T_out,
        hw_enc,  # (H/2, W/2)
        hw_out,  # (H, W)
        out_channels_pix=1,
        num_heads=4,
        window_size=(4, 4),
        use_temporal_attn: bool = True,
        use_spatial_attn: bool = True,
    ):
        super().__init__()

        self.T_out = T_out
        self.H_enc, self.W_enc = hw_enc
        self.H_out, self.W_out = hw_out
        self.out_channels_pix = out_channels_pix
        self.use_temporal_attn = use_temporal_attn
        self.use_spatial_attn = use_spatial_attn

        self.s1_cross_attn = TemporalCrossAttention(
            dim=enc_dim_stage1, num_heads=num_heads
        )
        self.s0_cross_attn = TemporalCrossAttention(
            dim=enc_dim_stage0, num_heads=num_heads
        )

        # Stage1 decoder blocks
        self.dec_s1_blocks = nn.ModuleList(
            [
                DualAttentionBlock(
                    dim=enc_dim_stage1,
                    num_heads=num_heads,
                    window_size=window_size,
                    use_temporal=use_temporal_attn,
                    use_spatial=use_spatial_attn,
                )
                for _ in range(2)
            ]
        )

        self.up_s1_to_s0 = Upsample3DLayer(
            dim=enc_dim_stage1,
            out_dim=enc_dim_stage0,
            target_size=(1, self.H_out, self.W_out),
            temporal_upsample=False,
            layout="THWC",
        )

        self.dec_s0_blocks = nn.ModuleList(
            [
                DualAttentionBlock(
                    dim=enc_dim_stage0 * 2,
                    num_heads=num_heads,
                    window_size=window_size,
                    use_temporal=use_temporal_attn,
                    use_spatial=use_spatial_attn,
                )
                for _ in range(2)
            ]
        )

        self.head = nn.Linear(enc_dim_stage0 * 2, out_channels_pix)

        self.future_t_embed = nn.Embedding(T_out, enc_dim_stage1)

    def forward(self, feat_s1, feat_s0):
        B, T_in, H1, W1, C1 = feat_s1.shape
        _, _, H0, W0, C0 = feat_s0.shape
        assert (
            H0 == self.H_out and W0 == self.W_out
        ), f"Decoder expected H_out={self.H_out}, W_out={self.W_out}; got {H0}, {W0}."
        assert (
            H1 == self.H_enc and W1 == self.W_enc
        ), f"Decoder expected encoder size {(self.H_enc, self.W_enc)}; got {(H1, W1)}."

        h_base = feat_s1[:, -1].unsqueeze(1)  # (B,1,H1,W1,C1)
        outputs = []

        for t in range(self.T_out):

            t_query = self.future_t_embed(
                torch.tensor(t, device=h_base.device)
            ).reshape(1, 1, 1, 1, -1)
            q_t = h_base + t_query

            h_t = self.s1_cross_attn(q_t, feat_s1)

            for blk in self.dec_s1_blocks:
                h_t = blk(h_t)

            h_up = self.up_s1_to_s0(h_t)  # (B,1,H0,W0,C0)

            dynamic_skip = self.s0_cross_attn(h_up, feat_s0)

            dec_s0 = torch.cat([h_up, dynamic_skip], dim=-1)

            for blk in self.dec_s0_blocks:
                dec_s0 = blk(dec_s0)

            B2, _, H2, W2, C2 = dec_s0.shape
            f = dec_s0.reshape(B2 * H2 * W2, C2)
            pix = self.head(f)  # (B2*H2*W2, C_pix)
            frame = pix.reshape(B, 1, H2, W2, self.out_channels_pix)

            outputs.append(frame)

            h_base = h_t

        return torch.cat(outputs, dim=1)  # (B,T_out,H0,W0,C_pix)


class PixelDualAttModel(nn.Module):

    def __init__(
        self,
        input_shape,
        target_shape,
        base_dim=64,
        stage1_dim=128,
        num_blocks_stage0=2,
        num_blocks_stage1=2,
        num_heads=4,
        window_size=(4, 4),
        attn_drop=0.0,
        ffn_drop=0.0,
        use_temporal_attn: bool = True,
        use_spatial_attn: bool = True,
    ):
        super().__init__()

        T_in, H, W, C_in = input_shape
        T_out, H2, W2, C_out = target_shape
        assert C_in == C_out, f"C_in={C_in} and C_out={C_out} must match."
        assert H == H2 and W == W2, "Input and output spatial resolutions must match."
        assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2."

        self.input_shape = input_shape
        self.target_shape = target_shape

        self.encoder = DualAttEncoder(
            in_channels=C_in,
            base_dim=base_dim,
            stage1_dim=stage1_dim,
            num_blocks_stage0=num_blocks_stage0,
            num_blocks_stage1=num_blocks_stage1,
            num_heads=num_heads,
            window_size=window_size,
            max_T=T_in,
            max_H=H,
            max_W=W,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
            use_temporal_attn=use_temporal_attn,
            use_spatial_attn=use_spatial_attn,
        )

        self.decoder = PixelDualAttDecoder(
            enc_dim_stage1=stage1_dim,
            enc_dim_stage0=base_dim,
            T_out=T_out,
            hw_enc=(H // 2, W // 2),
            hw_out=(H, W),
            out_channels_pix=C_in,
            num_heads=num_heads,
            window_size=window_size,
            use_temporal_attn=use_temporal_attn,
            use_spatial_attn=use_spatial_attn,
        )

    def forward(self, x):
        """
        x: (B, T_in, H, W, C_pix)
        """
        feat_s1, feat_s0 = self.encoder(x)
        y = self.decoder(feat_s1=feat_s1, feat_s0=feat_s0)
        return y
