"""
Transformer for EEG classification
"""

import argparse
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
from fvcore.nn import FlopCountAnalysis
cudnn.benchmark = False
cudnn.deterministic = True

from cp_trans import AC, set_seed, judge_done, DEVICE
from encoder import EncoderNet
from thop import profile

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


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, rank=6):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.d_k = emb_size // num_heads

        #self.keys = nn.Linear(emb_size, emb_size)
        #self.queries = nn.Linear(emb_size, emb_size)
        #self.values = nn.Linear(emb_size, emb_size)

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

        # q = torch.einsum('bqac,cr->bqar', x, self.W_Q2)
        # q = torch.einsum('bqar,ar->bqr', q, self.W_Q1)
        # q = torch.einsum('bqr,dr->bqd', q, self.W_Q0)
        q = torch.einsum('bqac, cr, ar, dr->bqd', [x, self.W_Q2, self.W_Q1, self.W_Q0])

        # k = torch.einsum('bkac,cr->bkar', x, self.W_K2)
        # k = torch.einsum('bkar,ar->bkr', k, self.W_K1)
        # k = torch.einsum('bkr,dr->bkd', k, self.W_K0)
        k = torch.einsum('bkac, cr, ar, dr->bkd', [x, self.W_K2, self.W_K1, self.W_K0])

        # v = torch.einsum('bvac,cr->bvar', x, self.W_V2)
        # v = torch.einsum('bvar,ar->bvr', v, self.W_V1)
        # v = torch.einsum('bvr,dr->bvd', v, self.W_V0)
        v = torch.einsum('bvac, cr, ar, dr->bvd', [x, self.W_V2, self.W_V1, self.W_V0])
        # queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        # values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
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
    def __init__(self, emb_size,
                       num_heads=5,
                       drop_p=0.5,
                        forward_expansion=4,
                        forward_drop_p=0.5):
        super().__init__()
        attn_res = ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                             MultiHeadAttention(emb_size, num_heads, drop_p),
                                             nn.Dropout(drop_p)))
        self.attn_res = attn_res
        ffn_res = ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                                            nn.Dropout(drop_p)))
        self.ffn_res = ffn_res


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


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
    def __init__(self, emb_size=10, depth=3, n_classes=4, **kwargs):
        super().__init__()
        # channel_attention(),
        residualAdd = ResidualAdd(nn.Sequential(nn.LayerNorm(1000),channel_attention(),nn.Dropout(0.5),))
        self.ResidualAdd = residualAdd

        patchEmbedding = PatchEmbedding(emb_size)
        self.PatchEmbedding = patchEmbedding

        transformerEncoder = TransformerEncoder(depth, emb_size)
        self.TransformerEncoder = transformerEncoder

        classificationHead = ClassificationHead(emb_size, n_classes)
        self.ClassificationHead = classificationHead


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
    def __init__(self, action_interval, floder, nsub, path=1):
        super(Trans, self).__init__()
        self.batch_size = 50
        self.unit_batch_size = 1
        self.n_epochs = action_interval
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

        self.log_write = open(floder + "log_subject%d.txt" % self.nSub, "w") if path else None

        self.img_shape = (self.channels, self.img_height, self.img_width)  # something no use

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = ViT().cuda()
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

        Score = 0
        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                score_list = []
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                # compute layer importance
                for block in self.model.module.TransformerEncoder:
                    grad = torch.autograd.grad(loss, block.attn_res.fn[1].parameters(), retain_graph=True)
                    tuples = zip(grad, block.attn_res.fn[1].parameters())
                    importence = list(map(lambda p: (p[0] * p[1]).pow(2).sum(), tuples))
                    score_list.append(sum(importence))
                score_list = torch.stack(score_list, dim=0)
                Score = Score + score_list

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

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
                    macs, params = get_model_complexity_info(self.model.module, (1, 16, 1000),
                                                                         as_strings=True,
                                                                         print_per_layer_stat=False, verbose=True)
                    flops = FlopCountAnalysis(self.model.module, torch.randn(1, 1, 16, 1000).cuda())
                    # flops_thop, _ = profile(self.model.module, inputs=(torch.randn(1, 1, 16, 1000).cuda(),))
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        Score = Score / (self.n_epochs * total_step)
        torch.save(self.model.module.state_dict(), './models/smartformer/model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        # self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        # self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred, Score, macs, params, flops.total()# , flops_thop

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



# action to decompose
def change_cpcores(model, action, reward, r_lim):
    rank0 = model.W_Q0.shape[1]
    rank = action + r_lim
    # input_size = model.W_Q.shape[0]
    if rank0 == rank or rank == 0:
        return rank0, reward

    Wq0 = model.W_Q0.cpu().detach().numpy()
    Wq1 = model.W_Q1.cpu().detach().numpy()
    Wq2 = model.W_Q2.cpu().detach().numpy()

    Wk0 = model.W_K0.cpu().detach().numpy()
    Wk1 = model.W_K1.cpu().detach().numpy()
    Wk2 = model.W_K2.cpu().detach().numpy()

    Wv0 = model.W_V0.cpu().detach().numpy()
    Wv1 = model.W_V1.cpu().detach().numpy()
    Wv2 = model.W_V2.cpu().detach().numpy()

    weight = np.array([1.]*rank0)
    wq = tl.cp_to_tensor((weight, [Wq0, Wq1, Wq2]))
    wk = tl.cp_to_tensor((weight, [Wk0, Wk1, Wk2]))
    wv = tl.cp_to_tensor((weight, [Wv0, Wv1, Wv2]))
    Wq = tl.tensor(wq)
    Wk = tl.tensor(wk)
    Wv = tl.tensor(wv)
    while True:
        try:
            weight1, WQ = parafac(Wq, rank)
            weight2, WK = parafac(Wk, rank)
            weight3, WV = parafac(Wv, rank)
            break
        except:
            reward += -0.005
            if rank == 1:

                model.W_Q0 = nn.Parameter(torch.tensor(Wq0, dtype=torch.float).to(DEVICE))
                model.W_Q1 = nn.Parameter(torch.tensor(Wq1, dtype=torch.float).to(DEVICE))
                model.W_Q2 = nn.Parameter(torch.tensor(Wq2, dtype=torch.float).to(DEVICE))

                model.W_K0 = nn.Parameter(torch.tensor(Wk0, dtype=torch.float).to(DEVICE))
                model.W_K1 = nn.Parameter(torch.tensor(Wk1, dtype=torch.float).to(DEVICE))
                model.W_K2 = nn.Parameter(torch.tensor(Wk2, dtype=torch.float).to(DEVICE))

                model.W_V0 = nn.Parameter(torch.tensor(Wv0, dtype=torch.float).to(DEVICE))
                model.W_V1 = nn.Parameter(torch.tensor(Wv1, dtype=torch.float).to(DEVICE))
                model.W_V2 = nn.Parameter(torch.tensor(Wv2, dtype=torch.float).to(DEVICE))

                return rank0, reward
            else:
                rank -= 1

    model.W_Q0 = nn.Parameter(torch.tensor(WQ[0], dtype=torch.float).to(DEVICE))
    model.W_Q1 = nn.Parameter(torch.tensor(WQ[1], dtype=torch.float).to(DEVICE))
    model.W_Q2 = nn.Parameter(torch.tensor(WQ[2], dtype=torch.float).to(DEVICE))

    model.W_K0 = nn.Parameter(torch.tensor(WK[0], dtype=torch.float).to(DEVICE))
    model.W_K1 = nn.Parameter(torch.tensor(WK[1], dtype=torch.float).to(DEVICE))
    model.W_K2 = nn.Parameter(torch.tensor(WK[2], dtype=torch.float).to(DEVICE))

    model.W_V0 = nn.Parameter(torch.tensor(WV[0], dtype=torch.float).to(DEVICE))
    model.W_V1 = nn.Parameter(torch.tensor(WV[1], dtype=torch.float).to(DEVICE))
    model.W_V2 = nn.Parameter(torch.tensor(WV[2], dtype=torch.float).to(DEVICE))

    return rank, reward


def step(model, encoder, action, last_averAcc, min_index, r_lim, reward_acc):
    reward = 0
    token = []
    real_ranks = []

    j = 0
    for block in model.model.module.TransformerEncoder:
        if j == min_index:
            real_rank, reward = change_cpcores(block.attn_res.fn[1], action, reward, r_lim)
        else:
            real_rank, reward = change_cpcores(block.attn_res.fn[1], -1, reward, r_lim)
        token += ['MultiHeadAttention-0-5-2-%d' % real_rank]
        real_ranks.append(real_rank)
        j += 1

    next_info, _ = encoder(token)
    bestAcc, averAcc, Y_true, Y_pred, Score, macs, params, flops = model.train()
    next_state = next_info[0][min_index]

    if last_averAcc - averAcc <= reward_acc: # the threshold for Acc
        reward += averAcc / last_averAcc
    else:
        reward += - last_averAcc / averAcc

    return next_info, next_state, bestAcc, averAcc, real_ranks[min_index], reward, Y_true, Y_pred, Score, macs, params, flops




def main(action_interval, seed, random_seed, epsilon, reward_acc, seed_list=None):

    if random_seed:
        model_floder = f'./models/smartformer/random_seed/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/'
        floder = f"./results/smartformer/random_seed/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/"
    else:
        seed_n = seed

        model_floder = f'./models/smartformer/seed_{seed}/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/'
        floder = f"./results/smartformer/seed_{seed_n}/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/"

    path = floder + 'sub_result.txt'

    if not os.path.exists(model_floder):
        os.makedirs(model_floder)

    if not os.path.exists(floder):
        os.makedirs(floder)

    best = 0
    aver = 0
    result_write = open(path, "w")

    start_time = time.time()
    Best = []
    for i in range(9):
    # for i in (3, 4, 5, 7):
    # if 1:
        # i = 2
        if random_seed:
            if seed_list is None:
                seed_n = np.random.randint(500)
            else:
                seed_n = seed_list[i]

            print('seed is ' + str(seed_n))
            random.seed(seed_n)
            np.random.seed(seed_n)
            torch.manual_seed(seed_n)
            torch.cuda.manual_seed(seed_n)
            torch.cuda.manual_seed_all(seed_n)

        else:
            set_seed(seed_n)

        print('Subject %d' % (i+1))
        n_epoch = 2000
        trans = Trans(action_interval, floder, i + 1)

        if i == 0:
            # Compute init rank params and Mac
            init_macs, init_params = get_model_complexity_info(trans.model.module, (1, 16, 1000), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
            # x = torch.randn(1, 1, 16, 1000).cuda()
            init_flops = FlopCountAnalysis(trans.model.module, torch.randn(1, 1, 16, 1000).cuda())
            # init_flops_thop, _ = profile(trans.model.module, inputs=(torch.randn(1, 1, 16, 1000).cuda(),))
            # result_write.write(f'Thop Init FLOPs: {init_flops_thop}, {init_flops_thop / 1e+9} G, {init_flops_thop / 1e+6} M' + '\n')
            result_write.write(f'Fvcore Init FLOPs: {init_flops.total()}, {init_flops.total() / 1e+9} G, {init_flops.total() / 1e+6} M' + '\n')
            result_write.write('{:<30}  {:<8}'.format('Init Computational complexity: ', init_macs) + "\n")
            result_write.write('{:<30}  {:<8}'.format('Init Number of parameters: ', init_params) + "\n")
            result_write.write('--------------------------------------------------------- ' + "\n")
            # unit processing time
            init_unit_forward_time, init_unit_backward_time, init_total_time = trans.unit_process_time()
            result_write.write(f'The init unit forward time is: {init_unit_forward_time:.5f} s' + "\n")
            result_write.write(f'The init unit backward time is: {init_unit_backward_time:.5f} s' + "\n")
            result_write.write(f'The init total time is: {init_total_time:.5f} s' + "\n")
            result_write.write('========================================================= ' + "\n")

        # encoder or decoder, n_head, dv, rank
        token_list = ['MultiHeadAttention-0-5-2-6'] * 3  # num of Attention
        result_write.write('初始状态：{}'.format(token_list) + '\n')
        enc = EncoderNet(512, 64)
        arch_info, _ = enc(token_list)
        # train for layer importance
        bestAcc, averAcc, Y_true, Y_pred, Score, macs, params, flops = trans.train()
        best_macs, best_params, best_flops = macs, params, flops
        last_averAcc = averAcc
        # chose the smallest layer_importance, decompose it only
        min_index = torch.argmin(Score).item()
        # create score mask
        score_mask = torch.tensor([0.]*3, device=DEVICE)
        done_mask = torch.tensor([1.]*3, device=DEVICE)
        state = arch_info[0][min_index]
        ac = AC(state, 6, epsilon)
        r_lim = 1    # the smallest R
        action_list = []
        rank_list = []
        critic_loss = []
        for j in range((n_epoch - trans.n_epochs) // trans.n_epochs):
            if score_mask.all() == done_mask.all():
                bestAcc_, averAcc_, Y_true, Y_pred, Score, macs, params, flops = trans.train()
            else:
                action = ac.choose_action(state)
                next_info, next_state, bestAcc_, averAcc_, real_ranks, reward, Y_true, Y_pred, Score, macs, params, flops = step(trans, enc, action,
                                                                                     last_averAcc, min_index, r_lim, reward_acc)
                loss_c, _, _ = ac.train_net(state, reward, next_state, action)
                state = next_state
                last_averAcc = averAcc_
                action_list.append(action)
                rank_list.append(real_ranks)
                critic_loss.append(loss_c.item())
                done = judge_done(action_list)
                if done:
                    # r_lim = rank_list[-1]
                    # ac.change_out(r_lim-1)
                    result_write.write(
                    'The {} layer R is ：{} 。Action is ：{}'.format(min_index + 1,rank_list,action_list) + '\n')
                    score_mask[min_index] = 1
                    score = Score.masked_fill(score_mask == 1, float('inf'))
                    min_index = torch.argmin(score).item()
                    state = next_info[0][min_index]
                    # ac = AC(state, 6, epsilon)
                    action_list = []
                    rank_list = []

            if bestAcc_ > bestAcc:
                bestAcc = bestAcc_
                best_macs, best_params, best_flops = macs, params, flops
                torch.save(trans.model.module, f'{model_floder}/sub{i+1}.pt')
            averAcc += averAcc_

        trans.log_write.write('critic loss: ' + str(critic_loss) + '\n')

        result_write.write(
        'The {} layer R is ：{} 。Action is ：{}'.format(min_index + 1, rank_list, action_list) + '\n')

        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('{}'.format(summary(trans.model, (1, 16, 1000))))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('**Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        result_write.write('--------------------------------------------------------- ' + "\n")
        # when model arrive its beat performance, compute the macs and params
        # result_write.write(f'Best Thop FLOPs: {best_thop_flops}, {best_thop_flops / 1e+9} G, {best_thop_flops / 1e+6} M' + '\n')
        result_write.write(f'Best Fvcore FLOPs: {best_flops}, {best_flops / 1e+9} G, {best_flops / 1e+6} M' + '\n')
        result_write.write('{:<30}  {:<8}'.format('Best Computational complexity: ', best_macs) + "\n")
        result_write.write('{:<30}  {:<8}'.format('Best Number of parameters: ', best_params) + "\n")
        result_write.write('--------------------------------------------------------- ' + "\n")
        # finish decompose and rl process, compute the final params and mac
        final_macs, final_params = get_model_complexity_info(trans.model.module, (1, 16, 1000), as_strings=True,
                                                       print_per_layer_stat=False, verbose=True)
        final_flops = FlopCountAnalysis(trans.model.module, torch.randn(1, 1, 16, 1000).cuda())
        # final_flops_thop, _ = profile(trans.model.module, inputs=(torch.randn(1, 1, 16, 1000).cuda(),))
        # result_write.write(
        # f'Final FLOPs: {final_flops_thop.total()}, {final_flops_thop.total() / 1e+9} G, {final_flops_thop.total() / 1e+6} M' + '\n')
        result_write.write(
        f'Final FLOPs: {final_flops.total()}, {final_flops.total() / 1e+9} G, {final_flops.total() / 1e+6} M' + '\n')
        result_write.write('{:<30}  {:<8}'.format('Final Computational complexity: ', final_macs) + "\n")
        result_write.write('{:<30}  {:<8}'.format('Final Number of parameters: ', final_params) + "\n")
        result_write.write('--------------------------------------------------------- ' + "\n")
        # unit processing time
        final_unit_forward_time, final_unit_backward_time, final_total_time = trans.unit_process_time()
        result_write.write(f'The final unit forward time is: {final_unit_forward_time:.5f} s' + "\n")
        result_write.write(f'The final unit backward time is: {final_unit_backward_time:.5f} s' + "\n")
        result_write.write(f'The final total time is: {final_total_time:.5f} s' + "\n")
        result_write.write('========================================================= ' + "\n")
        # plot_confusion_matrix(Y_true, Y_pred, i+1)
        Best.append(bestAcc)
        best = best + bestAcc
        aver = aver + averAcc
        
        '''
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
        '''
    
    end_time = time.time()
    best = best / 9
    aver = aver / 9
    Best = np.array(Best)
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.write(f'The running time is: {round(end_time - start_time, 2)} s \n')
    result_write.write('**The average Best accuracy is: ' + str(Best.mean()) + "\n")
    result_write.write('**The std Best accuracy is: ' + str(Best.std()) + "\n")
    result_write.close()


def unit_time(action_interval, seed, random_seed=0, epsilon=0.1, reward_acc=0.1):
    if random_seed:
        model_floder = f'./models/smartformer/random_seed/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/'
        floder = f"./results/smartformer/random_seed/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/"
    else:
        seed_n = seed
        set_seed(seed_n)
        model_floder = f'./models/smartformer/seed_{seed}/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/'
        floder = f"./results/smartformer/seed_{seed_n}/epsilon_{epsilon}/reward_acc_{reward_acc}/action_interval_{action_interval}/"

    path = floder + 'unit_time.txt'

    if not os.path.exists(model_floder):
        os.makedirs(model_floder)

    if not os.path.exists(floder):
        os.makedirs(floder)

    result_write = open(path, "w")
    trans = Trans(action_interval=action_interval, floder=floder, nsub=1, path=0)

    unit_forward_time, unit_backward_time, total_time = trans.unit_process_time()

    result_write.write(f'The unit forward time is: {unit_forward_time:.5f} s' + "\n")
    result_write.write(f'The unit backward time is: {unit_backward_time:.5f} s' + "\n")
    result_write.write(f'The total time is: {total_time:.5f} s' + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer family for EEG')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--random_seed', type=int, default=0, help='seed')
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon greedy')
    parser.add_argument('--reward_acc', type=float, default=0.1, help='reward acc')
    parser.add_argument('--action_interval', type=int, default=50, help='action interval for RL stage')
    args = parser.parse_args()

    # seed_list = [188, 322, 123, 448, 2021, 123, 120, 433, 448]
    main(args.action_interval, args.seed, args.random_seed, args.epsilon, args.reward_acc)
    unit_time(args.action_interval, args.seed, args.random_seed, args.epsilon, args.reward_acc)

