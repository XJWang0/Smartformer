import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
# from model import PositionalEncoding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_vocabulary = None

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

class Vocabulary:
    """
    store embedding code of each NAS module
    """
    def __init__(self, token_list):
        self.token_list = token_list
        self.vocab = {}
        for idx, token in enumerate(token_list):
            self.vocab[token] = idx
            self.vocab[idx] = token

    @property
    def size(self):
        return len(self.vocab) // 2

    def get_code(self, token_list):
        return [self.vocab[token] for token in token_list]

    def get_token(self, code_list):
        return [self.vocab[code] for code in code_list]

    def __str__(self):
        return str(self.vocab)

def get_vocabulary():
    global _vocabulary
    if _vocabulary is not None:
        return _vocabulary
    token_expand = 10
    token_list = []
    # token_list += ['transformer-%d-%d' % (i,j) for i in range(1,5) for j in range(1,5)] #Encoder 和 Decoder 的数量
    # token_list += ['encoder-%d' % i for i in range(1,11)] #encoder-block num
    # token_list += ['decoder-%d' % i for i in range(1,11)]
    # encoder or decoder, n_head, dv, rank
    token_list += ['MultiHeadAttention-%d-%d-%d-%d' % (i, j, k, m) for i in range(0, 1)
                   for j in range(1, 21) for k in range(1, 65) for m in range(13)]
    # encoder or decoder, n_head,mask,rank1,rank2
    # token_list += ['MultiHeadAttention-%d-%d-%d' % (i, j, k) for i in range(1, 21) for j in (0, 1) for k in range(128)]

    # token_list += ['MultiHeadAttention-%d-%d-%d' % (i, j, k) for i in range(1, 21) for j in (0, 1) for k in range(128)]
    _vocabulary = Vocabulary(token_list)
    return _vocabulary

class EncoderNet(nn.Module):
    """
    Encoder network similar to EAS
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_unit = torch.nn.LSTM(input_size, hidden_size, 1, True, True, bidirectional=True)
        self.embedding_unit = torch.nn.Embedding(get_vocabulary().size, input_size)
        self.pos_emb = PositionalEncoding(input_size)

        self.embedding_unit = self.embedding_unit.cuda()
        self.lstm_unit = self.lstm_unit.cuda()
        self.pos_emb = self.pos_emb.cuda()

    def embedding(self, token_list):
        vocab = get_vocabulary()
        codes = vocab.get_code(token_list)
        input = torch.tensor(codes).cuda()
        out = self.embedding_unit(input)

        return out.unsqueeze(0)

    def forward(self, token_list):
        """
        get encoder output
        :param token_list: list of token: list of str
        :return: output, (h_n, c_n) as described in lstm
        """
        embedding_tensor = self.embedding(token_list)
        embedding_tensor = self.pos_emb(embedding_tensor)
        output, (h_n, c_n) = self.lstm_unit(embedding_tensor)
        return output, (h_n, c_n)
