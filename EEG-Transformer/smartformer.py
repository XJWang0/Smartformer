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

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from common_spatial_pattern import csp
# from confusion_matrix import plot_confusion_matrix
# from cm_no_normal import plot_confusion_matrix_nn
# from torchsummary import summary

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

from cp_trans import AC, set_seed, judge_done, DEVICE
from encoder import EncoderNet


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

        q = torch.einsum('bqac,cr->bqar', x, self.W_Q2)
        q = torch.einsum('bqar,ar->bqr', q, self.W_Q1)
        q = torch.einsum('bqr,dr->bqd', q, self.W_Q0)

        k = torch.einsum('bkac,cr->bkar', x, self.W_K2)
        k = torch.einsum('bkar,ar->bkr', k, self.W_K1)
        k = torch.einsum('bkr,dr->bkd', k, self.W_K0)

        v = torch.einsum('bvac,cr->bvar', x, self.W_V2)
        v = torch.einsum('bvar,ar->bvr', v, self.W_V1)
        v = torch.einsum('bvr,dr->bvd', v, self.W_V0)

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
    def __init__(self, nsub):
        super(Trans, self).__init__()
        self.batch_size = 50
        self.n_epochs = 50
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.nSub = nsub
        self.start_epoch = 0
        self.root = r'./data/result/'  # the path of data

        self.pretrain = False

        self.log_write = open("results/log_subject%d.txt" % self.nSub, "w")

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
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        Score = Score / (self.n_epochs * total_step)
        torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred, Score


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


def step(model, encoder, action, last_averAcc, min_index, r_lim):
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
    bestAcc, averAcc, Y_true, Y_pred, Score = model.train()
    next_state = next_info[0][min_index]

    if last_averAcc - averAcc <= 0.1: # the threshold for Acc
        reward += averAcc / last_averAcc
    else:
        reward += - last_averAcc / averAcc

    return next_info, next_state, bestAcc, averAcc, real_ranks[min_index], reward, Y_true, Y_pred, Score




def main():
    seed_n = 448
    set_seed(seed_n)
    best = 0
    aver = 0
    result_write = open("results/sub_result.txt", "w")

    for i in range(9):
        print('Subject %d' % (i+1))
        n_epoch = 2000
        trans = Trans(i + 1)

        # encoder or decoder, n_head, dv, rank
        token_list = ['MultiHeadAttention-0-5-2-6'] * 3  # num of Attention
        result_write.write('初始状态：{}'.format(token_list) + '\n')
        enc = EncoderNet(512, 64)
        arch_info, (h, c) = enc(token_list)
        # train for layer importance
        bestAcc, averAcc, Y_true, Y_pred, Score = trans.train()
        last_averAcc = averAcc
        # chose the smallest layer_importance, decompose it only
        min_index = torch.argmin(Score).item()
        # create score mask
        score_mask = torch.tensor([0.]*3, device=DEVICE)
        done_mask = torch.tensor([1.]*3, device=DEVICE)
        state = arch_info[0][min_index]
        ac = AC(state, 6)
        r_lim = 1    # the smallest R
        action_list = []
        rank_list = []
        for j in range((n_epoch - trans.n_epochs) // trans.n_epochs):
            if score_mask.all() == done_mask.all():
                bestAcc_, averAcc_, Y_true, Y_pred, Score = trans.train()
            else:
                action = ac.choose_action(state)
                next_info, next_state, bestAcc_, averAcc_, real_ranks, reward, Y_true, Y_pred, Score = step(trans, enc, action,
                                                                                     last_averAcc, min_index, r_lim)
                ac.train_net(state, reward, next_state, action)
                state = next_state
                last_averAcc = averAcc_
                action_list.append(action)
                rank_list.append(real_ranks)
                done = judge_done(action_list)
                if done:
                    # r_lim = rank_list[-1]
                    # ac.change_out(r_lim-1)
                    result_write.write(
                    'The{}layer R is ：{} 。Action is ：{}'.format(min_index + 1,rank_list,action_list) + '\n')
                    score_mask[min_index] = 1
                    score = Score.masked_fill(score_mask == 1, float('inf'))
                    min_index = torch.argmin(score).item()
                    state = next_info[0][min_index]
                    action_list = []
                    rank_list = []

            if bestAcc_ > bestAcc:
                bestAcc = bestAcc_
            averAcc += averAcc_
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('{}'.format(summary(trans.model, (1, 16, 1000))))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('**Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        result_write.write('========================================================= ' + "\n")
        # plot_confusion_matrix(Y_true, Y_pred, i+1)
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    main()
    # trans = Trans(1)
    # print(trans.n_params)
    # print(trans.n_params1)
