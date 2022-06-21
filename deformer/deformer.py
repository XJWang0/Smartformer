import math
import torch

from torch import nn, Tensor
from typing import Optional
import copy

from cp_trans import Encoder, EncoderBlock

def generate_mask(n_pix):
    mask = torch.tril(torch.ones(2 * n_pix, 2 * n_pix))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def generate_multi_mask(n_pix):
    mask = torch.tril(torch.ones(2 * n_pix + 2, 2 * n_pix + 2))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


class DEformer(nn.Module):
    def __init__(
        self,
        img_size,
        pos_in_feats,
        pixel_n,
        mlp_layers,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        initrange = 0.1

        pos_mlp = nn.Sequential()
        pix_mlp = nn.Sequential()
        pix_in_feats = pos_in_feats + 1
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            pos_mlp.add_module(f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats))
            pix_mlp.add_module(f"layer{layer_idx}", nn.Linear(pix_in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                pix_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            pos_in_feats = out_feats
            pix_in_feats = out_feats

        self.pos_mlp = pos_mlp
        self.pix_mlp = pix_mlp

        d_model = mlp_layers[-1]
        self.d_model = d_model
        chns = 1 if pixel_n == 1 else 3
        n_pix = img_size ** 2 * chns
        self.register_buffer("mask", generate_mask(n_pix))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pixel_classifier = nn.Linear(d_model, pixel_n)
        self.pixel_classifier.weight.data.uniform_(-initrange, initrange)
        self.pixel_classifier.bias.data.zero_()

    def forward(self, tensors):
        device = list(self.pixel_classifier.parameters())[0].device

        positions = tensors["positions"].to(device)
        b = positions.size(0)

        pos_feats = self.pos_mlp(positions) * math.sqrt(self.d_model)

        pixels = tensors["pixels"].unsqueeze(2).to(device)
        pos_pixels = torch.cat([positions, pixels], dim=2)
        # pixels = tensors["pixels"].unsqueeze(1).to(device)
        # pos_pixels = torch.cat([positions, pixels], dim=1)
        pix_feats = self.pix_mlp(pos_pixels) * math.sqrt(self.d_model)
        '''
        combined = torch.zeros(2 * len(pos_feats), self.d_model).to(device)
        combined[::2] = pos_feats
        combined[1::2] = pix_feats
        '''
        combined = torch.zeros(b, 2 * pos_feats.size(1), self.d_model).to(device)

        combined[:, ::2] = pos_feats
        combined[:, 1::2] = pix_feats

        # outputs = self.transformer(combined.unsqueeze(1), self.mask)
        # preds = self.pixel_classifier(outputs.squeeze(1)[::2])

        outputs = self.transformer(combined.transpose(0, 1), self.mask)
        outputs = outputs.transpose(0, 1)
        aa = outputs[:, ::2]
        aaa = outputs[:, ::2, :]
        preds = self.pixel_classifier(outputs[:, ::2])

        return preds


class CP_DEformer(nn.Module):
    def __init__(
        self,
        # rank,
        img_size,
        pos_in_feats,
        pixel_n,
        mlp_layers,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        initrange = 0.1

        pos_mlp = nn.Sequential()
        pix_mlp = nn.Sequential()
        pix_in_feats = pos_in_feats + 1
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            pos_mlp.add_module(f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats))
            pix_mlp.add_module(f"layer{layer_idx}", nn.Linear(pix_in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                pix_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            pos_in_feats = out_feats
            pix_in_feats = out_feats

        self.pos_mlp = pos_mlp
        self.pix_mlp = pix_mlp

        d_model = mlp_layers[-1]
        self.d_model = d_model
        chns = 1 if pixel_n == 1 else 3
        n_pix = img_size ** 2 * chns
        self.register_buffer("mask", generate_mask(n_pix))
        encoder_layer = EncoderBlock(360,
            d_model, nhead, dim_feedforward, True, dropout
        )
        self.transformer = Encoder(encoder_layer, num_layers)

        self.pixel_classifier = nn.Linear(d_model, pixel_n)
        self.pixel_classifier.weight.data.uniform_(-initrange, initrange)
        self.pixel_classifier.bias.data.zero_()

    def forward(self, tensors):
        device = list(self.pixel_classifier.parameters())[0].device

        positions = tensors["positions"].to(device)
        b = positions.size(0)

        pos_feats = self.pos_mlp(positions) * math.sqrt(self.d_model)

        pixels = tensors["pixels"].unsqueeze(2).to(device)
        pos_pixels = torch.cat([positions, pixels], dim=2)

        pix_feats = self.pix_mlp(pos_pixels) * math.sqrt(self.d_model)

        combined = torch.zeros(b, 2 * pos_feats.size(1), self.d_model).to(device)
        combined[:, ::2] = pos_feats
        combined[:, 1::2] = pix_feats

        outputs = self.transformer(combined.transpose(0, 1), self.mask)
        outputs = outputs.transpose(0, 1)

        preds = self.pixel_classifier(outputs[:, ::2])

        return preds



# =================================================================================
class Encoder_(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm:Optional=None):
        super(Encoder_, self).__init__()
        encoder_layer1 = EncoderBlock(430, d_model, nhead, dim_feedforward, True, dropout)
        encoder_layer2 = EncoderBlock(290, d_model, nhead, dim_feedforward, True, dropout)

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer1) for i in range(4)])
        self.layers.append(encoder_layer2)
        self.layers.append(encoder_layer2)

        # self.num_layers = num_layers
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


class CP_DEformer_(nn.Module):
    def __init__(
        self,
        # rank,
        img_size,
        pos_in_feats,
        pixel_n,
        mlp_layers,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        initrange = 0.1

        pos_mlp = nn.Sequential()
        pix_mlp = nn.Sequential()
        pix_in_feats = pos_in_feats + 1
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            pos_mlp.add_module(f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats))
            pix_mlp.add_module(f"layer{layer_idx}", nn.Linear(pix_in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                pix_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            pos_in_feats = out_feats
            pix_in_feats = out_feats

        self.pos_mlp = pos_mlp
        self.pix_mlp = pix_mlp

        d_model = mlp_layers[-1]
        self.d_model = d_model
        chns = 1 if pixel_n == 1 else 3
        n_pix = img_size ** 2 * chns
        self.register_buffer("mask", generate_mask(n_pix))

        self.transformer = Encoder_(d_model, nhead, dim_feedforward, dropout)

        self.pixel_classifier = nn.Linear(d_model, pixel_n)
        self.pixel_classifier.weight.data.uniform_(-initrange, initrange)
        self.pixel_classifier.bias.data.zero_()

    def forward(self, tensors):
        '''
        device = list(self.pixel_classifier.parameters())[0].device

        positions = tensors["positions"].to(device)
        pos_feats = self.pos_mlp(positions) * math.sqrt(self.d_model)
        pixels = tensors["pixels"].unsqueeze(1).to(device)
        pos_pixels = torch.cat([positions, pixels], dim=1)
        pix_feats = self.pix_mlp(pos_pixels) * math.sqrt(self.d_model)

        combined = torch.zeros(2 * len(pos_feats), self.d_model).to(device)
        combined[::2] = pos_feats
        combined[1::2] = pix_feats

        outputs = self.transformer(combined.unsqueeze(1), self.mask)
        preds = self.pixel_classifier(outputs.squeeze(1)[::2])
        '''

        # '''
        device = list(self.pixel_classifier.parameters())[0].device

        positions = tensors["positions"].to(device)
        b = positions.size(0)

        pos_feats = self.pos_mlp(positions) * math.sqrt(self.d_model)

        pixels = tensors["pixels"].unsqueeze(2).to(device)
        pos_pixels = torch.cat([positions, pixels], dim=2)

        pix_feats = self.pix_mlp(pos_pixels) * math.sqrt(self.d_model)

        combined = torch.zeros(b, 2 * pos_feats.size(1), self.d_model).to(device)
        combined[:, ::2] = pos_feats
        combined[:, 1::2] = pix_feats

        outputs = self.transformer(combined.transpose(0, 1), self.mask)
        outputs = outputs.transpose(0, 1)

        preds = self.pixel_classifier(outputs[:, ::2])
        #'''

        return preds


class DEformer_(nn.Module):
    def __init__(
        self,
        img_size,
        pos_in_feats,
        pixel_n,
        mlp_layers,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        initrange = 0.1

        pos_mlp = nn.Sequential()
        pix_mlp = nn.Sequential()
        pix_in_feats = pos_in_feats + 1
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            pos_mlp.add_module(f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats))
            pix_mlp.add_module(f"layer{layer_idx}", nn.Linear(pix_in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                pix_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            pos_in_feats = out_feats
            pix_in_feats = out_feats

        self.pos_mlp = pos_mlp
        self.pix_mlp = pix_mlp

        d_model = mlp_layers[-1]
        self.d_model = d_model
        chns = 1 if pixel_n == 1 else 3
        n_pix = img_size ** 2 * chns
        self.register_buffer("mask", generate_mask(n_pix))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pixel_classifier = nn.Linear(d_model, pixel_n)
        self.pixel_classifier.weight.data.uniform_(-initrange, initrange)
        self.pixel_classifier.bias.data.zero_()

    def forward(self, tensors):
        device = list(self.pixel_classifier.parameters())[0].device

        positions = tensors["positions"].to(device)
        pos_feats = self.pos_mlp(positions) * math.sqrt(self.d_model)
        pixels = tensors["pixels"].unsqueeze(1).to(device)
        pos_pixels = torch.cat([positions, pixels], dim=1)
        pix_feats = self.pix_mlp(pos_pixels) * math.sqrt(self.d_model)

        combined = torch.zeros(2 * len(pos_feats), self.d_model).to(device)
        combined[::2] = pos_feats
        combined[1::2] = pix_feats

        outputs = self.transformer(combined.unsqueeze(1), self.mask)
        preds = self.pixel_classifier(outputs.squeeze(1)[::2])

        return preds

''' 
device = list(self.pixel_classifier.parameters())[0].device

        positions = tensors["positions"].to(device)
        pos_feats = self.pos_mlp(positions) * math.sqrt(self.d_model)
        pixels = tensors["pixels"].unsqueeze(1).to(device)
        pos_pixels = torch.cat([positions, pixels], dim=1)
        pix_feats = self.pix_mlp(pos_pixels) * math.sqrt(self.d_model)

        combined = torch.zeros(2 * len(pos_feats), self.d_model).to(device)
        combined[::2] = pos_feats
        combined[1::2] = pix_feats

        outputs = self.transformer(combined.unsqueeze(1), self.mask)
        preds = self.pixel_classifier(outputs.squeeze(1)[::2])

        return preds'''
