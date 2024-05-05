import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os
from enum import Enum

class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        if attn is not None:
            print(f'attn shape is {attn.shape}'+'\n')

        return self.out_projection(out), attn


class CP_AutoCorrelationLayer(nn.Module):
    def __init__(self, rank, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(CP_AutoCorrelationLayer, self).__init__()

        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model

        self.inner_correlation = correlation


        self.W_Q0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_Q1 = nn.Parameter(torch.randn((n_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_Q2 = nn.Parameter(torch.randn((self.d_keys, rank), dtype=torch.float), requires_grad=True)

        self.W_K0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_K1 = nn.Parameter(torch.randn((n_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_K2 = nn.Parameter(torch.randn((self.d_keys, rank), dtype=torch.float), requires_grad=True)

        self.W_V0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_V1 = nn.Parameter(torch.randn((n_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_V2 = nn.Parameter(torch.randn((self.d_values, rank), dtype=torch.float), requires_grad=True)

        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        self._reset_param()

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.compute(queries, self.W_Q0, self.W_Q1, self.W_Q2).view(B, L, H, -1)
        keys = self.compute(keys, self.W_K0, self.W_K1, self.W_K2).view(B, S, H, -1)
        values = self.compute(values, self.W_V0, self.W_V1, self.W_V2).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

    def compute(self, x, f0, f1, f2):
        x = x.view(x.size(0), x.size(1), self.n_heads, -1)

        # out = torch.einsum('blnd, dr->blnr', x, f2)
        # out = torch.einsum('blnr, nr->blr', out, f1)
        # out = torch.einsum('blr, mr->blm', out, f0)
        out = torch.einsum('blnd, dr, nr, mr->blm', [x, f2, f1, f0])

        return out

    def _reset_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

class SVD_AutoCorrelationLayer(nn.Module):
    def __init__(self, rank, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(SVD_AutoCorrelationLayer, self).__init__()

        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model

        self.inner_correlation = correlation


        self.W_Q0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_Q1 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)

        self.W_K0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_K1 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)

        self.W_V0 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)
        self.W_V1 = nn.Parameter(torch.randn((d_model, rank), dtype=torch.float), requires_grad=True)

        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        self._reset_param()

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.compute(queries, self.W_Q0, self.W_Q1).view(B, L, H, -1)
        keys = self.compute(keys, self.W_K0, self.W_K1).view(B, S, H, -1)
        values = self.compute(values, self.W_V0, self.W_V1).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

    def compute(self, x, f0, f1):

        # out = torch.einsum('blf, fr->blr', x, f1)
        # out = torch.einsum('blr, mr->blm', out, f0)
        
        out = torch.einsum('blf, fr, mr->blm', [x, f1, f0])

        return out

    def _reset_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class Lin_AutoCorrelationLayer(nn.Module):
    def __init__(self, seq_len, k, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(Lin_AutoCorrelationLayer, self).__init__()
        self.seq_len = seq_len
        self.k = k
        
        self.emb_size = d_model
        self.num_heads = n_heads
        self.dim_head = d_model // n_heads


        self.inner_correlation = correlation
        self.queries = nn.Linear(d_model, d_model)

        self.keys = nn.Linear(d_model, d_model)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.values = nn.Linear(d_model, d_model)
        self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))
        
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        d_h, h, k = self.dim_head, self.num_heads, self.k
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        
        kv_len = S
        # assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.queries(queries)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        keys = self.keys(keys)
        values = self.values(values)

        kv_projs = (self.proj_k, self.proj_v)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(B, L, h, -1)

        merge_key_values = lambda t: t.reshape(B, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))
        keys, values = keys.transpose(1, 2), values.transpose(1, 2)
        
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

        
class Colla_AutoCorrelationLayer(nn.Module):
    def __init__(self,
        dim_input: int,
        dim_value_all: int,
        dim_key_query_all: int,
        dim_output: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        output_attentions: bool, 
        use_dense_layer: bool,
        mixing_initialization: MixingMatrixInit = MixingMatrixInit.UNIFORM,
    ):
        super(Colla_AutoCorrelationLayer, self).__init__()

        # save args
        self.dim_input = dim_input
        self.dim_value_all = dim_value_all  # the dim of value for all heads
        self.dim_key_query_all = dim_key_query_all  # the dim of query and key for all heads
        self.dim_output = dim_output
        self.num_attention_heads = num_attention_heads
        self.output_attentions = output_attentions
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.mixing_initialization = mixing_initialization
        self.use_dense_layer = use_dense_layer

        self.dim_value_per_head = dim_value_all // num_attention_heads
        self.attention_head_size = (
                dim_key_query_all / num_attention_heads
        )  # does not have to be integer

        # intialize parameters
        self.query = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.key = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.value = nn.Linear(dim_input, dim_value_all)

        self.content_bias = nn.Linear(dim_input, num_attention_heads, bias=False)

        self.mixing = self.init_mixing_matrix()

        self.dense = (
            nn.Linear(dim_value_all, dim_output) if use_dense_layer else nn.Sequential()
        )

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape


        query_layer = self.query(queries)
        key_layer = self.key(keys)

        # point wise multiplication of the mixing coefficient per head with the shared query projection
        # (batch, from_seq, dim) x (head, dim) -> (batch, head, from_seq, dim)
        mixed_query = query_layer[..., None, :, :] * self.mixing[..., :, None, :]

        # broadcast the shared key for all the heads
        # (batch, 1, to_seq, dim)
        mixed_key = key_layer[..., None, :, :]

        value_layer = self.value(values)
        value_layer = self.transpose_for_scores(value_layer)
        
        # (batch, head, from_seq, to_seq)
        attention_scores = torch.matmul(mixed_query, mixed_key.transpose(-1, -2))

        # add the content bias term
        # (batch, to_seq, heads)
        content_bias = self.content_bias(keys)
        # (batch, heads, 1, to_seq)
        broadcast_content_bias = content_bias.transpose(-1, -2).unsqueeze(-2)
        attention_scores += broadcast_content_bias

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim_value_all,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.dense(context_layer)

        return context_layer, attention_probs
       

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def init_mixing_matrix(self, scale=0.2):
        mixing = torch.zeros(self.num_attention_heads, self.dim_key_query_all)

        if self.mixing_initialization is MixingMatrixInit.CONCATENATE:
            # last head will be smaller if not equally divisible
            dim_head = int(math.ceil(self.dim_key_query_all / self.num_attention_heads))
            for i in range(self.num_attention_heads):
                mixing[i, i * dim_head : (i + 1) * dim_head] = 1.0

        elif self.mixing_initialization is MixingMatrixInit.ALL_ONES:
            mixing.one_()
        elif self.mixing_initialization is MixingMatrixInit.UNIFORM:
            mixing.normal_(std=scale)
        else:
            raise ValueError(
                "Unknown mixing matrix initialization: {}".format(
                    self.mixing_initialization
                )
            )

        return nn.Parameter(mixing)


class BTD_AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout, d_ff, output_attention, d_keys=None,
                 d_values=None, is_sum=0):
        super(BTD_AutoCorrelationLayer, self).__init__()
        self.is_sum = is_sum
        
        self.d_head = d_head if d_head is not None else d_model//n_heads
        self.n_heads = n_heads
        
        self.query_projection = nn.Linear(d_model, self.d_head * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_head * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_head * n_heads)
        
        self.core_nums = n_heads
        self.R = self.d_head
        self.core_value = nn.Parameter(F.softmax(torch.FloatTensor(self.core_nums, self.R), dim=-1), requires_grad=True)
        
        self.drop = nn.Dropout(dropout)
        self.out_projection = nn.Linear(d_ff, d_model)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H, d = self.n_heads, self.d_head
        
        scaling = (H*d) ** (1 / 2)
        
        queries = self.query_projection(queries).view(H, B, L, -1)
        keys = self.key_projection(keys).view(H, B, S, -1)
        values = self.value_projection(values).view(H, B, S, -1)
        
        full_matrixs = 0
        for i in range(self.core_nums):
            full_matrix_1 = torch.einsum('h, bih,bjh,bkh->bijk',
                                         [self.core_value[i], queries[i], keys[i], values[i]]).contiguous()

            full_matrix_1 = full_matrix_1.view(B, L, -1)
            
            full_matrixs += (full_matrix_1)

        # linear projection
        full_matrixs.mul_(1 / self.core_nums)
        # print(f'full_matrixs shape is : {full_matrixs.shape}')
        out = self.out_projection(full_matrixs)
        
        energy = torch.einsum('bqd, bkd -> bqk', queries.reshape(B, L, -1), keys.reshape(B, S, -1))  # batch, num_heads, query_len, key_len

        
        attn = F.softmax(energy / scaling, dim=-1)
        
        if self.output_attention:
            return out, attn
        else:
            return out, None
