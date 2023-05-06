import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as Data
import math
import copy
from typing import List, Tuple, Optional,Union, Callable
import tensorly as tl
import numpy as np
from tensorly.decomposition import parafac


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  Sin PE
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# MultiHead Attention no decomposition version
class MultiHeadAttention(nn.Module):
    def __init__(self, rank, d_model, num_heads, dropout=0.1, batch_first=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.rank = rank

        self.keys = nn.Linear(d_model, d_model, bias=False)
        self.queries = nn.Linear(d_model, d_model, bias=False)
        self.values = nn.Linear(d_model, d_model, bias=False)

        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.batch_first = batch_first

    def ScaledDotProductAttention(self, q, k, v, mask, padding_mask, bsz, tgt_len, src_len):
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d_k)

        if mask is not None:   # mask: [seq_len, seq_len]
          mask = mask.unsqueeze(0)
          scores += mask

        if padding_mask is not None:
            assert padding_mask.shape == (bsz, src_len)
            scores = scores.view(bsz, self.num_heads, tgt_len, src_len)
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            scores = scores.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(scores, -1)

        if self.dropout_p > 0.0:
            attn_output_weights = self.drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return attn_output, attn_output_weights

    def forward(self, query, key, value, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:
        # x:[seq_len,batch,featrues]

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, batch_size, f = query.size()
        src_len = key.size(0)

        q = self.queries(query)
        k = self.keys(key)
        v = self.values(value)

        Q = q.view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)
        K = k.view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)
        V = v.view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)

        attn_output, attn_output_weights = self.ScaledDotProductAttention(Q, K, V, mask, padding_mask, batch_size, tgt_len, src_len)
        # attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)
        # linear again
        attn_output = self.W_O(attn_output)
        # attn_output = attn_output.transpose(0, 1)
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


# MultiHead Attention decomposition version
class Decomp_MultiHeadAttention(nn.Module):
    def __init__(self, rank, d_model, num_heads, dropout = 0.1, batch_first=False):
        super(Decomp_MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.rank = rank
        # W = tl.tensor(np.random.randn(d_model, self.num_heads, self.d_k) / (d_model + self.num_heads + self.d_k))
        # weight, factor = parafac(W, rank)

        self.W_Q0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_Q1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_Q2 = nn.Parameter(torch.randn((self.d_k, rank), dtype=torch.float), requires_grad=True)

        self.W_K0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_K1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_K2 = nn.Parameter(torch.randn((self.d_k, rank), dtype=torch.float), requires_grad=True)

        self.W_V0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_V1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_V2 = nn.Parameter(torch.randn((self.d_k, rank), dtype=torch.float), requires_grad=True)

        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.batch_first = batch_first

        self._reset_parameters()

    def ScaledDotProductAttention(self, q, k, v, mask, padding_mask, bsz, tgt_len, src_len):
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d_k)

        if mask is not None:  # mask: [seq_len, seq_len]
            mask = mask.unsqueeze(0)
            scores += mask

        if padding_mask is not None:
            assert padding_mask.shape == (bsz, src_len)
            scores = scores.view(bsz, self.num_heads, tgt_len, src_len)
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            scores = scores.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(scores, -1)

        if self.dropout_p > 0.0:
            attn_output_weights = self.drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return attn_output, attn_output_weights

    def forward(self, query, key, value, mask: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None)\
            -> Tuple[Tensor, Tensor]:
        # x:[seq_len,batch,featrues]

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, batch_size, f = query.size()
        src_len = key.size(0)

        query = query.reshape(batch_size, -1, self.num_heads, self.d_k)
        key = key.reshape(batch_size, -1, self.num_heads, self.d_k)
        value = value.reshape(batch_size, -1, self.num_heads, self.d_k)

        q = torch.einsum('bqac,cr->bqar', query, self.W_Q2)
        q = torch.einsum('bqar,ar->bqr', q, self.W_Q1)
        q = torch.einsum('bqr,dr->bqd', q, self.W_Q0)

        k = torch.einsum('bkac,cr->bkar', key, self.W_K2)
        k = torch.einsum('bkar,ar->bkr', k, self.W_K1)
        k = torch.einsum('bkr,dr->bkd', k, self.W_K0)

        v = torch.einsum('bvac,cr->bvar', value, self.W_V2)
        v = torch.einsum('bvar,ar->bvr', v, self.W_V1)
        v = torch.einsum('bvr,dr->bvd', v, self.W_V0)

        Q = q.view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)
        K = k.view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)
        V = v.view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)

        attn_output, attn_output_weights = self.ScaledDotProductAttention(Q, K, V, mask, padding_mask, batch_size, tgt_len, src_len)
        # attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)
        # linear again
        attn_output = self.W_O(attn_output)
        # attn_output = attn_output.transpose(0, 1)
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights

    def _reset_parameters(self):
        # Initiate parameters in the transformer model
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)



class EncoderBlock(nn.Module):
    def __init__(self, rank, d_model, num_heads, dim_feedforward, decomp, dropout=0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]]=F.relu,
                 batch_first=False, norm_first=False):
        super(EncoderBlock, self).__init__()
        if decomp:
            self.attn = Decomp_MultiHeadAttention(rank=rank, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                                  batch_first=batch_first)
        else:
            self.attn = MultiHeadAttention(rank=rank, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                           batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, src, src_mask: Optional[Tensor]=None, src_padding_mask: Optional[Tensor]=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, padding_mask):
        x = self.attn(x, x, x, attn_mask, padding_mask)[0]
        x = self.drop1(x)
        return x

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.drop(self.activation(self.linear1(x))))
        x = self.drop2(x)
        return x

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class DecoderBlock(nn.Module):
    def __init__(self, rank, d_model, num_heads, dim_feedforward, dropout=0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]]=F.relu,
                 batch_first=False, norm_first=False, decomp=True):
        super(DecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        if decomp:
            self.mask_attn = Decomp_MultiHeadAttention(rank=rank, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                                       batch_first=batch_first)
            self.attn = Decomp_MultiHeadAttention(rank=rank, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                                  batch_first=batch_first)
        else:
            self.mask_attn = MultiHeadAttention(rank=rank, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                                 batch_first=batch_first)
            self.attn = MultiHeadAttention(rank=rank, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                           batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, tgt, memory, tgt_mask: Optional[Tensor]=None, tgt_padding_mask: Optional[Tensor]=None,
                memory_mask: Optional[Tensor]=None, memory_padding_mask: Optional[Tensor]=None):
        """
        tgt: [batch_size, tgt_len, d_model]
        memory: [batch_size, src_len, d_model]
        tgt_mask: [batch_size, tgt_len, tgt_len]
        tgt_padding_mask: [batch_size, tgt_len]
        memory_mask: [batch_size, src_len, src_len]
        memory_padding_mask: [batch_size, src_len]
        """
        x = tgt
        if self.norm_fist:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, padding_mask):
        x = self.mask_attn(x, x, x, attn_mask, padding_mask)[0]
        x = self.dropout1(x)
        return x

    # memory attention block
    def _mha_block(self, x, mem, attn_mask, padding_mask):
        x = self.attn(x, mem, mem, attn_mask, padding_mask)[0]
        x = self.drop2(x)
        return x

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout3(x)
        return x

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm:Optional=None):
        super(Encoder, self).__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None):

        output = src
        for block in self.layers:
            output = block(output, mask, padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm:Optional=None):
        super(Decoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask: Optional[Tensor]=None, tgt_padding_mask: Optional[Tensor]=None,
                memory_mask: Optional[Tensor]=None, memory_padding_mask: Optional[Tensor]=None):
        """
        tgt: [batch_size, tgt_len]
        memory: [batch_size, src_len, d_model]   # used in Encoder-Decoder Attention
        """
        output = tgt

        for block in self.layers:
            output = block(output, memory, tgt_mask, tgt_padding_mask, memory_mask, memory_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(nn.Module):
    def __init__(self, rank, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, batch_first=False, norm_first=False, decomp=True):
        super(Transformer, self).__init__()
        encoder_layer = EncoderBlock(rank=rank, d_model=d_model, dim_feedforward=dim_feedforward,
                                     num_heads=nhead, dropout=dropout, activation=activation,
                                     batch_first=batch_first, norm_first=norm_first, decomp=decomp)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = Encoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

        decoder_layer = DecoderBlock(rank=rank, d_model=d_model, dim_feedforward=dim_feedforward,
                                     num_heads=nhead, dropout=dropout, activation=activation,
                                     batch_first=batch_first, norm_first=norm_first, decomp=decomp)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        self._reset_parameters()

    def forward(self, src, tgt, src_mask: Optional[Tensor]=None, tgt_mask: Optional[Tensor]=None,
                src_padding_mask: Optional[Tensor]=None, tgt_padding_mask: Optional[Tensor]=None,
                memory_mask: Optional[Tensor]=None, memory_padding_mask: Optional[Tensor]=None):
        """
        src: [src_len, batch_size]
        tgt: [tgt_len, batch_size]
        """
        memory = self.encoder(src, src_mask, src_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, tgt_padding_mask, memory_mask, memory_padding_mask)

        return output

    def _reset_parameters(self):
        # Initiate parameters in the transformer model
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)