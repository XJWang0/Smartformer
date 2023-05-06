import torch
import torch.nn.functional as F
import torch.nn as nn
from enum import Enum
import math


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Lin_Attention(nn.Module):
    def __init__(self, k, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, seq_len=64):
        super(Lin_Attention, self).__init__()
        self.seq_len = seq_len
        self.k = k

        self.emb_size = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

        self.queries = nn.Linear(dim, dim, bias=False)

        self.keys = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.values = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        self.n_heads = num_heads
        self.scale = self.dim_head ** -0.5

    def forward(self, x):
        d_h, h, k = self.dim_head, self.num_heads, self.k
        B, L, _ = x.shape
        _, S, _ = x.shape

        kv_len = S
        # assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.queries(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        keys = self.keys(x)
        values = self.values(x)

        kv_projs = (self.proj_k, self.proj_v)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        q = queries.reshape(B, h, L, -1)

        merge_key_values = lambda t: t.reshape(B, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        k, v = map(merge_key_values, (keys, values))
        # k, v = keys.transpose(1, 2), values.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3


class Colla_Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int=8,
                 attention_dropout: float=0.1,
                 projection_dropout=0.1,
                 use_dense_layer: bool=True,
                 mixing_initialization: MixingMatrixInit = MixingMatrixInit.UNIFORM,
                 ):
        super(Colla_Attention, self).__init__()

        # save args
        self.dim_input = dim
        self.dim_value_all = dim  # the dim of value for all heads
        self.dim_key_query_all = dim  # the dim of query and key for all heads
        self.dim_output = dim
        self.num_attention_heads = num_heads
        # nself.output_attentions = output_attentions
        self.attention_probs_dropout_prob = attention_dropout
        self.mixing_initialization = mixing_initialization
        self.use_dense_layer = use_dense_layer

        self.dim_value_per_head = dim // num_heads
        self.attention_head_size = (
                dim / num_heads
        )  # does not have to be integer

        # intialize parameters
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

        self.content_bias = nn.Linear(dim, num_heads, bias=False)

        self.mixing = self.init_mixing_matrix()

        self.dense = (
            nn.Linear(dim, dim) if use_dense_layer else nn.Sequential()
        )

        self.dropout = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, L, _ = x.shape
        _, S, _ = x.shape


        query_layer = self.query(x)
        key_layer = self.key(x)

        # point wise multiplication of the mixing coefficient per head with the shared query projection
        # (batch, from_seq, dim) x (head, dim) -> (batch, head, from_seq, dim)
        mixed_query = query_layer[..., None, :, :] * self.mixing[..., :, None, :]

        # broadcast the shared key for all the heads
        # (batch, 1, to_seq, dim)
        mixed_key = key_layer[..., None, :, :]

        value_layer = self.value(x)
        value_layer = self.transpose_for_scores(value_layer)

        # (batch, head, from_seq, to_seq)
        attention_scores = torch.matmul(mixed_query, mixed_key.transpose(-1, -2))

        # add the content bias term
        # (batch, to_seq, heads)
        content_bias = self.content_bias(x)
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
        x = self.proj_drop(context_layer)

        return x


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


class BTD_Attention(nn.Module):
    def __init__(self, dim, num_heads, attention_dropout, projection_dropout, core_num, d_head=64, d_ff=64*64):
        super(BTD_Attention, self).__init__()

        self.d_head = d_head
        self.n_heads = core_num

        self.query_projection = nn.Linear(dim, self.d_head * core_num)
        self.key_projection = nn.Linear(dim, self.d_head * core_num)
        self.value_projection = nn.Linear(dim, self.d_head * core_num)

        self.core_nums = core_num
        self.R = self.d_head
        self.core_value = nn.Parameter(F.softmax(torch.FloatTensor(self.core_nums, self.R), dim=-1), requires_grad=True)

        self.drop = nn.Dropout(projection_dropout)

        self.out_projection = nn.Linear(d_ff, dim)


    def forward(self, x):
        B, L, _ = x.shape
        H, d = self.n_heads, self.d_head

        scaling = (H * d) ** (1 / 2)

        queries = self.query_projection(x).view(H, B, L, -1)
        keys = self.key_projection(x).view(H, B, L, -1)
        values = self.value_projection(x).view(H, B, L, -1)

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
        out = self.drop(out)
        # energy = torch.einsum('bqd, bkd -> bqk', queries, keys)  # batch, num_heads, query_len, key_len

        # attn = F.softmax(energy / scaling, dim=-1)

        return out