import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from .stochastic_depth import DropPath
import torch.nn as nn



class CP_Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, is_decomposed, rank, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.is_decomposed = is_decomposed
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = Linear(dim, dim * 3, bias=False)
        self.W_Q0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
        self.W_Q1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_Q2 = nn.Parameter(torch.randn((head_dim, rank), dtype=torch.float), requires_grad=True)

        self.W_K0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
        self.W_K1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_K2 = nn.Parameter(torch.randn((head_dim, rank), dtype=torch.float), requires_grad=True)

        self.W_V0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
        self.W_V1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
        self.W_V2 = nn.Parameter(torch.randn((head_dim, rank), dtype=torch.float), requires_grad=True)
        self._reset_param()

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = self.qkv(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def qkv(self, x):
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, -1)

        # q = torch.einsum('blnd, dr->blnr', x, self.W_Q2)
        # q = torch.einsum('blnr, nr->blr', q, self.W_Q1)
        # q = torch.einsum('blr, mr->blm', q, self.W_Q0).reshape(B, self.num_heads, L, -1)
        q = torch.einsum('blnd, dr, nr, mr->blm', [x, self.W_Q2, self.W_Q1, self.W_Q0]).reshape(B, self.num_heads, L,
                                                                                                -1)

        # k = torch.einsum('blnd, dr->blnr', x, self.W_K2)
        # k = torch.einsum('blnr, nr->blr', k, self.W_K1)
        # k = torch.einsum('blr, mr->blm', k, self.W_K0).reshape(B, self.num_heads, L, -1)
        k = torch.einsum('blnd, dr, nr, mr->blm', [x, self.W_K2, self.W_K1, self.W_K0]).reshape(B, self.num_heads, L,
                                                                                                -1)
        # v = torch.einsum('blnd, dr->blnr', x, self.W_V2)
        # v = torch.einsum('blnr, nr->blr', v, self.W_V1)
        # v = torch.einsum('blr, mr->blm', v, self.W_V0).reshape(B, self.num_heads, L, -1)
        v = torch.einsum('blnd, dr, nr, mr->blm', [x, self.W_V2, self.W_V1, self.W_V0]).reshape(B, self.num_heads, L,
                                                                                                -1)
        return q, k, v

    def _reset_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


class SVD_Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, is_decomposed, rank, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.is_decomposed = is_decomposed
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = Linear(dim, dim * 3, bias=False)
        self.W_Q0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
        self.W_Q1 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)

        self.W_K0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
        self.W_K1 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)

        self.W_V0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
        self.W_V1 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
        self._reset_param()

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = self.qkv(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def qkv(self, x):
        B, L, _ = x.shape
        # q = torch.einsum('bln, nr->blr', x, self.W_Q1)
        # q = torch.einsum('blr, mr->blm', q, self.W_Q0).reshape(B, self.num_heads, L, -1)
        q = torch.einsum('bln, nr, mr->blm', [x, self.W_Q1, self.W_Q0]).reshape(B, self.num_heads, L, -1)

        # k = torch.einsum('bln, nr->blr', x, self.W_K1)
        # k = torch.einsum('blr, mr->blm', k, self.W_K0).reshape(B, self.num_heads, L, -1)
        k = torch.einsum('bln, nr, mr->blm', [x, self.W_K1, self.W_K0]).reshape(B, self.num_heads, L, -1)

        # v = torch.einsum('bln, nr->blr', x, self.W_V1)
        # v = torch.einsum('blr, mr->blm', v, self.W_V0).reshape(B, self.num_heads, L, -1)
        v = torch.einsum('bln, nr, mr->blm', [x, self.W_V1, self.W_V0]).reshape(B, self.num_heads, L, -1)

        return q, k, v

    def _reset_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


class Smart_Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, is_decomposed, rank, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.is_decomposed = is_decomposed
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        if rank == 0:
            assert self.is_decomposed == 0
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)

            self.W_Q0 = None
            self.W_Q1 = None
            self.W_Q2 = None

            self.W_K0 = None
            self.W_K1 = None
            self.W_K2 = None

            self.W_V0 = None
            self.W_V1 = None
            self.W_V2 = None
        else:
            assert self.is_decomposed == 1
            self.q = None
            self.k = None
            self.v = None

            self.W_Q0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
            self.W_Q1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
            self.W_Q2 = nn.Parameter(torch.randn((head_dim, rank), dtype=torch.float), requires_grad=True)

            self.W_K0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
            self.W_K1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
            self.W_K2 = nn.Parameter(torch.randn((head_dim, rank), dtype=torch.float), requires_grad=True)

            self.W_V0 = nn.Parameter(torch.randn((dim, rank), dtype=torch.float), requires_grad=True)
            self.W_V1 = nn.Parameter(torch.randn((num_heads, rank), dtype=torch.float), requires_grad=True)
            self.W_V2 = nn.Parameter(torch.randn((head_dim, rank), dtype=torch.float), requires_grad=True)
            self._reset_param()

        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def _reset_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        B, N, C = x.shape
        if self.is_decomposed:
            q, k, v = self.to_qkv(x)
        else:
            q, k, v = self.qkv(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def qkv(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, self.num_heads, N, -1)
        k = self.k(x).reshape(B, self.num_heads, N, -1)
        v = self.v(x).reshape(B, self.num_heads, N, -1)
        return (q, k, v)

    def to_qkv(self, x):
        B, L, C = x.shape
        x = x.reshape(B, L, self.num_heads, -1)

        # q = torch.einsum('blnd, dr->blnr', x, self.W_Q2)
        # q = torch.einsum('blnr, nr->blr', q, self.W_Q1)
        # q = torch.einsum('blr, mr->blm', q, self.W_Q0).reshape(B, self.num_heads, L, -1)
        q = torch.einsum('blnd, dr, nr, mr->blm', [x, self.W_Q2, self.W_Q1, self.W_Q0]).reshape(B, self.num_heads, L,
                                                                                                -1)

        # k = torch.einsum('blnd, dr->blnr', x, self.W_K2)
        # k = torch.einsum('blnr, nr->blr', k, self.W_K1)
        # k = torch.einsum('blr, mr->blm', k, self.W_K0).reshape(B, self.num_heads, L, -1)
        k = torch.einsum('blnd, dr, nr, mr->blm', [x, self.W_K2, self.W_K1, self.W_K0]).reshape(B, self.num_heads, L,
                                                                                                -1)
        # v = torch.einsum('blnd, dr->blnr', x, self.W_V2)
        # v = torch.einsum('blnr, nr->blr', v, self.W_V1)
        # v = torch.einsum('blr, mr->blm', v, self.W_V0).reshape(B, self.num_heads, L, -1)
        v = torch.einsum('blnd, dr, nr, mr->blm', [x, self.W_V2, self.W_V1, self.W_V0]).reshape(B, self.num_heads, L,
                                                                                                -1)

        return (q, k, v)

    def move_qkv(self):
        # The q, k, v is decomposed, so move them
        try:
            assert self.is_decomposed == 0
            self.q = None
            self.k = None
            self.v = None
            self.is_decomposed = 1
        except:
            pass

    def move_cpcores(self):
        try:
            assert self.is_decomposed == 1
            self.W_Q0 = None
            self.W_Q1 = None
            self.W_Q2 = None

            self.W_K0 = None
            self.W_K1 = None
            self.W_K2 = None

            self.W_V0 = None
            self.W_V1 = None
            self.W_V2 = None
            self.is_decomposed = 0
        except:
            pass


class MaskedAttention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, theta, model_type, is_decomposed, rank, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        if theta > 0.:
            self.self_attn = Smart_Attention(is_decomposed=is_decomposed, rank=rank, dim=d_model, num_heads=nhead,
                                             attention_dropout=attention_dropout, projection_dropout=dropout)
        else:
            if model_type == 'svd':
                self.self_attn = SVD_Attention(is_decomposed=is_decomposed, rank=rank, dim=d_model, num_heads=nhead,
                                              attention_dropout=attention_dropout, projection_dropout=dropout)
            else:
                self.self_attn = CP_Attention(is_decomposed=is_decomposed, rank=rank, dim=d_model, num_heads=nhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout)


        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class MaskedTransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MaskedAttention(dim=d_model, num_heads=nhead,
                                         attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, mask=None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class CP_TransformerClassifier(Module):
    def __init__(self, theta,
                 model_type, rank, is_decomposed,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(theta=theta, model_type=model_type, is_decomposed=is_decomposed, rank=rank,
                                    d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class MaskedTransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='sine',
                 seq_len=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert seq_len is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            seq_len += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                seq_len += 1  # padding idx
                self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(seq_len,
                                                                          embedding_dim,
                                                                          padding_idx=True),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            MaskedTransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, mask=None):
        if self.positional_emb is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(size=(mask.shape[0], 1), device=mask.device), mask.float()], dim=1)
                mask = (mask > 0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim, padding_idx=False):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        if padding_idx:
            return torch.cat([torch.zeros((1, 1, dim)), pe], dim=1)
        return pe
