"""
Transformer for EEG classification
"""


import os
import numpy as np
import math
import random
import time
import scipy.io
import tensorly as tl
from tensorly.decomposition import parafac
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import argparse

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torchstat import stat
from ptflops import get_model_complexity_info
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from common_spatial_pattern import csp
# from confusion_matrix import plot_confusion_matrix
# from cm_no_normal import plot_confusion_matrix_nn
# from torchsummary import summary

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from enum import Enum
from fvcore.nn import FlopCountAnalysis
from cp_trans import set_seed
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('/home/syh/Documents/MI/code/Trans/TensorBoardX/')

# torch.cuda.set_device(6)
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (16, 5), stride=(1, 5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # position
        # x += self.positions
        return x

# CP-Attention, we set R=d_head
class MultiHeadAttention(nn.Module):
    def __init__(self, rank, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.d_k = emb_size // num_heads
        rank = self.d_k if rank is None else rank
        W = tl.tensor(np.random.randn(emb_size, self.num_heads, self.d_k) / (emb_size + self.num_heads + self.d_k))
        weight, factor = parafac(W, rank)

        self.W_Q0 = nn.Parameter(torch.tensor(factor[0], dtype=torch.float))
        self.W_Q1 = nn.Parameter(torch.tensor(factor[1], dtype=torch.float))
        self.W_Q2 = nn.Parameter(torch.tensor(factor[2], dtype=torch.float))
        # nn.init.xavier_normal(self.W_Q)
        self.W_K0 = nn.Parameter(torch.tensor(factor[0], dtype=torch.float))
        self.W_K1 = nn.Parameter(torch.tensor(factor[1], dtype=torch.float))
        self.W_K2 = nn.Parameter(torch.tensor(factor[2], dtype=torch.float))
        # nn.init.xavier_normal(self.W_K)
        self.W_V0 = nn.Parameter(torch.tensor(factor[0], dtype=torch.float))
        self.W_V1 = nn.Parameter(torch.tensor(factor[1], dtype=torch.float))
        self.W_V2 = nn.Parameter(torch.tensor(factor[2], dtype=torch.float))

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, tgt_len, f = x.size()
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)

        q = torch.einsum('bqac,cr->bqar', x, self.W_Q2)
        q = torch.einsum('bqar,ar->bqr', q, self.W_Q1)
        q = torch.einsum('bqr,dr->bqd', q, self.W_Q0)

        k = torch.einsum('bkac,cr->bkar', x, self.W_K2)
        k = torch.einsum('bkar,ar->bkr', k, self.W_K1)
        k = torch.einsum('bkr,dr->bkd', k, self.W_K0)

        v = torch.einsum('bvac,cr->bvar', x, self.W_V2)
        v = torch.einsum('bvar,ar->bvr', v, self.W_V1)
        v = torch.einsum('bvr,dr->bvd', v, self.W_V0)

        queries = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class SVD_MultiHeadAttention(nn.Module):
    def __init__(self, rank, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.d_k = emb_size // num_heads
        rank = self.d_k if rank is None else rank
        W = tl.tensor(np.random.randn(emb_size, self.num_heads * self.d_k) / (emb_size + self.num_heads * self.d_k))
        weight, factor = parafac(W, rank)

        self.W_Q0 = nn.Parameter(torch.tensor(factor[0], dtype=torch.float))
        self.W_Q1 = nn.Parameter(torch.tensor(factor[1], dtype=torch.float))
        # nn.init.xavier_normal(self.W_Q)
        self.W_K0 = nn.Parameter(torch.tensor(factor[0], dtype=torch.float))
        self.W_K1 = nn.Parameter(torch.tensor(factor[1], dtype=torch.float))
        # nn.init.xavier_normal(self.W_K)
        self.W_V0 = nn.Parameter(torch.tensor(factor[0], dtype=torch.float))
        self.W_V1 = nn.Parameter(torch.tensor(factor[1], dtype=torch.float))

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, tgt_len, f = x.size()

        q = torch.einsum('bqa,ar->bqr', x, self.W_Q1)
        q = torch.einsum('bqr,dr->bqd', q, self.W_Q0)

        k = torch.einsum('bka,ar->bkr', x, self.W_K1)
        k = torch.einsum('bkr,dr->bkd', k, self.W_K0)

        v = torch.einsum('bva,ar->bvr', x, self.W_V1)
        v = torch.einsum('bvr,dr->bvd', v, self.W_V0)

        queries = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, method, rank,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        if method == 'cp':
            Attention = MultiHeadAttention(rank, emb_size, num_heads, drop_p)
        elif method == 'svd':
            Attention = SVD_MultiHeadAttention(rank, emb_size, num_heads, drop_p)
        else:
            Attention = MultiHeadAttention(rank, emb_size, num_heads, drop_p)
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                Attention,
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, method, rank, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(method, rank, emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out


class ViT(nn.Sequential):
    def __init__(self, method, rank, emb_size=10, depth=3, n_classes=4, **kwargs):
        super().__init__(
            # channel_attention(),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(1000),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),

            PatchEmbedding(emb_size),
            TransformerEncoder(method, rank, depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class channel_attention(nn.Module):
    def __init__(self, sequence_num=1000, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(16, 16),
            # nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(16, 16),
            # nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out


class Trans():
    def __init__(self, method, rank, floder, nsub, path=1):
        super(Trans, self).__init__()
        self.batch_size = 50
        self.unit_batch_size = 1
        self.n_epochs = 2000
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.nSub = nsub
        self.start_epoch = 0
        self.root = './data/result/'  # the path of data

        self.pretrain = False

        self.log_write = open(f"{floder}/log_subject%d.txt" % self.nSub, "w") if path else None

        self.img_shape = (self.channels, self.img_height, self.img_width)  # something no use

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = ViT(method, rank).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        summary(self.model, (1, 16, 1000))

        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.n_params1 = sum([p.nelement() for p in self.model.parameters()])
        self.centers = {}

    def get_source_data(self):

        # to get the data of target subject
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        # test data
        # to get the data of target subject
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        # self.train_data = self.train_data[250:1000, :, :]
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # Mix the train and test data - a quick way to get start
        # But I agree, just shuffle data is a bad measure
        # You could choose cross validation, or get more data from more subjects, then Leave one subject out
        all_data = np.concatenate((self.allData, self.testData), 0)
        all_label = np.concatenate((self.allLabel, self.testLabel), 0)
        all_shuff_num = np.random.permutation(len(all_data))
        all_data = all_data[all_shuff_num]
        all_label = all_label[all_shuff_num]

        self.allData = all_data[:516]
        self.allLabel = all_label[:516]
        self.testData = all_data[516:]
        self.testLabel = all_label[516:]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        tmp_alldata = np.transpose(np.squeeze(self.allData), (0, 2, 1))
        Wb = csp(tmp_alldata, self.allLabel-1)  # common spatial pattern
        self.allData = np.einsum('abcd, ce -> abed', self.allData, Wb)
        self.testData = np.einsum('abcd, ce -> abed', self.testData, Wb)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Do some data augmentation is a potential way to improve the generalization ability
    def aug(self, img, label):
        aug_data = []
        aug_label = []
        return aug_data, aug_label

    def train(self):


        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)


        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr
        # some better optimization strategy is worthy to explore. Sometimes terrible over-fitting.


        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # out_epoch = time.time()

            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Test loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', acc)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred
        floder = './models/cpformer/'
        if not os.path.exists(floder):
            os.makedirs(floder)
        torch.save(self.model.module.state_dict(), floder + 'model.pth')

        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred

    def unit_process_time(self):
        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.unit_batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.unit_batch_size,
                                                           shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        
        forward_time = []
        backward_time = []
        total_time = []
        
        self.model.train()
        for i, (img, label) in enumerate(self.dataloader):
            img = Variable(img.cuda().type(self.Tensor))
            label = Variable(label.cuda().type(self.LongTensor))

            inference_time_1 = time.time()
            tok, outputs = self.model(img)
            inference_time_2 = time.time()
            loss = self.criterion_cls(outputs, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            inference_time_3 = time.time()
            
            unit_forward_time = inference_time_2 - inference_time_1
            unit_backward_time = inference_time_3 - inference_time_2
            unit_total_time = inference_time_3 - inference_time_1
            
            forward_time.append(unit_forward_time)
            backward_time.append(unit_backward_time)
            total_time.append(unit_total_time)
            
            if i == 99:
                break

        return (sum(forward_time)/len(forward_time)), (sum(backward_time)/len(backward_time)), (sum(total_time)/len(total_time))


def main(method, R):
    set_seed(123)
    best = 0
    aver = 0
    floder = f"results/{method}former/R_{R}/"
    path = floder + 'sub_result.txt'

    if not os.path.exists(floder):
        os.makedirs(floder)

    result_write = open(path, "w")
    rank = R
    start_time = time.time()
    for i in range(9):
    # i = 0
    # if (i+1):
        '''
        seed_n = np.random.randint(500)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        '''
        print('Subject %d' % (i+1))
        trans = Trans(method, rank, floder, i + 1)
        bestAcc, averAcc, Y_true, Y_pred = trans.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        # result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('**Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        # plot_confusion_matrix(Y_true, Y_pred, i+1)
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

    end_time = time.time()

    unit_forward_time, unit_backward_time, total_time = trans.unit_process_time()

    best = best / 9
    aver = aver / 9
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.write('========================================================= ' + "\n")
    result_write.write(f'The running time is: {round(end_time - start_time, 2)} s \n')
    # compute params, FLOPs, etc
    macs, params = get_model_complexity_info(trans.model.module, (1, 16, 1000), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    flops = FlopCountAnalysis(trans.model.module, torch.randn(1, 1, 16, 1000).cuda())
    result_write.write(f'FLOPs: {flops.total()}, {flops.total() / 1e+9} G, {flops.total() / 1e+6} M' + '\n')
    result_write.write('{:<30}  {:<8}'.format('Computational complexity: ', macs) + "\n")
    result_write.write('{:<30}  {:<8}'.format('Number of parameters: ', params) + "\n")

    result_write.write(f'The unit forward time is: {unit_forward_time:.5f} s' + "\n")
    result_write.write(f'The unit backward time is: {unit_backward_time:.5f} s' + "\n")
    result_write.write(f'The total time is: {total_time:.5f} s' + "\n")
    result_write.close()

def unit_time(method, R):
    floder = f"results/{method}former/R_{R}/"
    path = floder + 'unit_time.txt'

    if not os.path.exists(floder):
        os.makedirs(floder)

    result_write = open(path, "w")
    
    trans = Trans(method=method, rank=R, floder=None, nsub=1, path=0)
    
    unit_forward_time, unit_backward_time, total_time = trans.unit_process_time()
    macs, params = get_model_complexity_info(trans.model.module, (1, 16, 1000), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    flops = FlopCountAnalysis(trans.model.module, torch.randn(1, 1, 16, 1000).cuda())
    result_write.write(f'FLOPs: {flops.total()}, {flops.total() / 1e+9} G, {flops.total() / 1e+6} M' + '\n')
    result_write.write('{:<30}  {:<8}'.format('Computational complexity: ', macs) + "\n")
    result_write.write('{:<30}  {:<8}'.format('Number of parameters: ', params) + "\n")
    result_write.write('-----------------------------------------------------------------------\n')
    result_write.write(f'The unit forward time is: {unit_forward_time:.5f} s' + "\n")
    result_write.write(f'The unit backward time is: {unit_backward_time:.5f} s' + "\n")
    result_write.write(f'The total time is: {total_time:.5f} s' + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer family for EEG')
    parser.add_argument('--R', type=int, default=6, help='rank ')
    parser.add_argument('--decomp_method', type=str, default='cp', help='cp or svd ')

    args = parser.parse_args()
    main(args.decomp_method, args.R)
    unit_time(args.decomp_method, args.R)

