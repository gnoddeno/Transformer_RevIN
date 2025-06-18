from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Dropout, LayerNorm, Linear, Module, Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.transformer import _get_activation_fn, TransformerDecoder

from models.tranformers.transformer import TotalEmbedding


# =============================================================================
#                         [ 새로 추가된 구성 요소 ]
# =============================================================================
class RevIN(nn.Module):
    """
    리버서블 인스턴스 정규화
    ---------------------------------------------
    - mode='norm'  : 입력을 (배치·특성 단위)로 정규화
    - mode='denorm': 저장해 둔 통계치를 이용해 원 스케일 복원
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))
        self.register_buffer("_mean", None)  # 정규화 시 평균 저장
        self.register_buffer("_std",  None)  # 정규화 시 표준편차 저장

    def forward(self, x: Tensor, mode: str = "norm") -> Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std  = x.std (dim=1, keepdim=True)
            x_norm = (x - self._mean) / (self._std + self.eps)
            return self.gamma * x_norm + self.beta
        elif mode == "denorm":
            if self._mean is None or self._std is None:
                raise RuntimeError("먼저 mode='norm' 으로 호출해야 복원이 가능합니다.")
            x_denorm = (x - self.beta) / (self.gamma + self.eps)
            return x_denorm * (self._std + self.eps) + self._mean
        else:
            raise ValueError("mode는 'norm' 혹은 'denorm' 이어야 합니다.")


class PatchEmbedding(nn.Module):
    """
    1-D 패치 임베딩
    ---------------------------------------------
    - 시계열을 패치 단위로 분할(unfold)
    - 각 패치를 선형 투영하여 embed_dim 크기로 변환
    """
    def __init__(self, embed_dim: int, patch_len: int = 16, stride: int = 16):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(embed_dim * patch_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, L, E = x.shape
        if L < self.patch_len:
            raise ValueError("입력 길이는 patch_len 이상이어야 합니다.")
        n_patches = 1 + (L - self.patch_len) // self.stride
        patches = x.unfold(1, self.patch_len, self.stride)     # [B, n, patch_len, E]
        patches = patches.contiguous().view(B, n_patches, -1)  # [B, n, patch_len*E]
        return self.proj(patches)                              # [B, n, E]


class ChannelAttention(nn.Module):
    """
    채널 어텐션(SE 블록 형태)
    ---------------------------------------------
    - 시간 축 전역 평균 풀링 → 두 개의 FC → 시그모이드
    - 각 채널 중요도 가중치 부여
    """
    def __init__(self, embed_dim: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(embed_dim // reduction, embed_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        w = x.mean(dim=1)             # [B, E]
        w = self.fc2(self.relu(self.fc1(w)))
        w = self.sigmoid(w).unsqueeze(1)  # [B, 1, E]
        return x * w                   # 브로드캐스트 곱


class RMSNorm(Module):
    """
    RMSNorm : 루트 평균 제곱 기반 정규화
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)
# -----------------------------------------------------------------------------


class ConvolutionalMultiheadAttention(Module):
    """
    Conv + 다중 헤드 어텐션
    ---------------------------------------------
    1) 1-D 컨볼루션으로 국소 특성 추출
    2) 추출 결과를 표준 MHA에 투입
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, kernel_size=3, dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim은 num_heads로 나누어 떨어져야 합니다."

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

        self.conv  = nn.Conv1d(embed_dim, embed_dim, kernel_size, stride=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size, stride=1)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super().__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor] = None, need_weights: bool = True,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        # 컨볼루션 전처리
        query = query.transpose(2, 1)
        key   = key.transpose(2, 1)
        query = self.conv (F.pad(query, (self.kernel_size - 1, 0)))
        key   = self.conv2(F.pad(key,   (self.kernel_size - 1, 0)))
        query = query.transpose(2, 0).transpose(2, 1)
        key   = key  .transpose(2, 0).transpose(2, 1)
        value = value.transpose(1, 0)

        # 표준 MHA
        if not self._qkv_same_embed_dim:
            attn_output, attn_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask)

        return (attn_output.transpose(1, 0), attn_weights) if self.batch_first else (attn_output, attn_weights)


class TransformerEncoderLayerWithConvolutionalAttention(Module):
    """
    컨볼루션-어텐션 인코더 레이어
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, kernel_size=3,
                 dropout=0.1, activation="relu", layer_norm_eps=1e-5,
                 batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = ConvolutionalMultiheadAttention(
            d_model, nhead, kernel_size=kernel_size, dropout=dropout,
            batch_first=batch_first, **factory_kwargs)

        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = RMSNorm(d_model)  # RMSNorm 사용
        self.norm2 = RMSNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerDecoderLayerWithConvolutionalAttention(Module):
    """
    컨볼루션-어텐션 디코더 레이어
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5,
                 batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = ConvolutionalMultiheadAttention(d_model, nhead, dropout=dropout,
                                                         batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = ConvolutionalMultiheadAttention(d_model, nhead, dropout=dropout,
                                                              batch_first=batch_first, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


# =============================================================================
#                        최종 시계열 Transformer 모델
# =============================================================================
class TransformerRevIN(nn.Module):
    """
    RevIN + Patch + ChannelAttention + Conv-Attention Transformer
    """
    def __init__(self, d_model: int, input_features_count: int,
                 num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, kernel_size: int,
                 dropout: float, attention_heads: int):
        super().__init__()
        factory_kwargs = {'device': 'cuda'}

        # ----- 인코더 / 디코더 스택 구성 ---------------------------------------
        encoder_layer = TransformerEncoderLayerWithConvolutionalAttention(
            d_model, attention_heads, dim_feedforward, kernel_size,
            dropout, 'relu', 1e-5, True, **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayerWithConvolutionalAttention(
            d_model, attention_heads, dim_feedforward, dropout,
            'relu', 1e-5, True, **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.transformer = nn.Transformer(
            d_model, attention_heads, num_encoder_layers, num_decoder_layers,
            custom_encoder=encoder, custom_decoder=decoder,
            batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout)

        # ----- 임베딩 및 추가 모듈 ---------------------------------------------
        self.encoder_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout)
        self.decoder_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout)

        self.revin  = RevIN(input_features_count)
        self.patch_embed_enc = PatchEmbedding(d_model, patch_len=kernel_size, stride=kernel_size)
        self.patch_embed_dec = PatchEmbedding(d_model, patch_len=kernel_size, stride=kernel_size)
        self.patch_s = PatchEmbedding(d_model, patch_len = 8, stride = 4)
        self.patch_l = PatchEmbedding(d_model, patch_len = 32, stride = 16)
        self.channel_attn = ChannelAttention(d_model)
        self.rms_norm = RMSNorm(d_model)

        self.fuse = nn.Linear(d_model*2, d_model)

        self.projection = nn.Linear(d_model, 1, bias=True)
        self.relu = nn.ReLU()

    def _causal_mask(self, size: int, device):
        return torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)

    def forward(self, x_enc, x_dec, src_mask=None, tgt_mask=None):
        # ---------------------------------
        # 2. 임베딩
        # ---------------------------------
        enc_emb = self.encoder_embedding(x_enc)    # [B, 168, d]
        dec_emb = self.decoder_embedding(x_dec)    # [B, 120, d]  (디코더는 그대로)

        # ---------------------------------
        # 2-1. 다중 패치 인코딩 (인코더 전용)
        # ---------------------------------
        enc_s = self.patch_s(enc_emb)              # 예) [B, 41, d]
        enc_l = self.patch_l(enc_emb)              # 예) [B,  9, d]

        min_len = min(enc_s.size(1), enc_l.size(1))  # 9
        enc_s, enc_l = enc_s[:, :min_len], enc_l[:, :min_len]

        enc_s = self.rms_norm(self.channel_attn(enc_s))          # [B, 9, d]
        enc_l = self.rms_norm(self.channel_attn(enc_l))          # [B, 9, d]

        enc_emb = self.fuse(torch.cat([enc_s, enc_l], dim=-1))  # [B, 9, d]
        enc_emb = self.rms_norm(enc_emb)            # (선택) 마지막 한 번 더 Norm

        # ---------------------------------
        # 3. tgt_mask 크기 자동 보정
        # ---------------------------------
        if tgt_mask is not None and tgt_mask.size(0) != dec_emb.size(1):
            tgt_mask = self._causal_mask(dec_emb.size(1), dec_emb.device)

        # ---------------------------------
        # 4. Transformer
        # ---------------------------------
        out = self.transformer(enc_emb, dec_emb,
                               src_mask=src_mask, tgt_mask=tgt_mask)  # [B, 120, d]

        # ---------------------------------
        # 5. 1-채널 projection
        # ---------------------------------
        out = self.projection(self.relu(out))               # [B, 120, 1]

        # ---------------------------------
        # 6. 역정규화 (target feature만)
        # ---------------------------------           # [B, 120, 1]

        return out.squeeze(-1)                              # [B, 120]
