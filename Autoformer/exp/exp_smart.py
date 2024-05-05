from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import Autoformer, Smart_Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis
import os
import time

import warnings
import matplotlib.pyplot as plt

from cp_trans import AC, EncoderNet, DEVICE, judge_done
import tensorly as tl
from tensorly.decomposition import parafac

warnings.filterwarnings('ignore')

done = False

Score = torch.tensor([0.] * 3, device=DEVICE)
action_list = []
rank_list = []
state = 0
next_state = 0
next_info = 0
min_index = float('inf')
action = -1
reward = 0
last_train_loss = 1
action_train_loss = 0
action_interval = 100
log = ''

class frame():
    def __init__(self, args):
        # AC and decomposition
        # self.score_mask = torch.tensor([0.] * 3).cuda()
        self.done_mask = torch.tensor([1.] * 3).cuda()
        self.args = args
        self.action_interval = args.action_interval
        
        if self.args.RL:
            self.score_mask = torch.tensor([0.] * 3).cuda()
            # svd 256; cp 450
            token_list = ['MultiHeadAttention-0-8-64-%d' % self.args.rank] * 2
            token_list += ['MultiHeadAttention-1-8-64-%d' % self.args.rank]

            self.enc = EncoderNet(512, 64)
            self.arch_info = self.get_init_state(token_list)

            S = self.arch_info[0][0]
            # action dim, 512x512; if CP: 512x512=Rx(512+8+64); if SVD: 512x512=Rx(512+512)
            # CP: 28; SVD: 22
            action_dim = self.args.action_dim
            self.ac = AC(S, action_dim, args.epsilon)
        else:
            self.score_mask = torch.tensor([1.] * 3).cuda()

    def get_init_state(self, token_list):
        out, _ = self.enc(token_list)
        return out
    
    def reset_ac(self, state):
        self.ac = AC(state, self.args.action_dim, self.args.epsilon)

class CP_Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(CP_Exp_Main, self).__init__(args)
        self.args = args
        self.frame = frame(args)
        
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'CP_Autoformer': Smart_Autoformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def get_input_shape(self):
        data, unit_loader = self._get_data(flag='unit')

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(unit_loader):
            batch_x = batch_x.float().to(self.device)

            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            break

        return batch_x, batch_x_mark, dec_inp, batch_y_mark

    def get_input(self, input_shape):
        batch_x, batch_x_mark, dec_inp, batch_y_mark = self.get_input_shape()
        input = {
            'x_enc': batch_x,
            'x_mark_enc': batch_x_mark,
            'x_dec': dec_inp,
            'x_mark_dec': batch_y_mark,
        }
        return input

    def compute_params_macs(self):
        # compute params, FLOPs, etc
        macs, params = get_model_complexity_info(self.model, (1,), as_strings=0, input_constructor=self.get_input,
                                                 print_per_layer_stat=False, verbose=True)
        x = self.get_input_shape()
        flops = FlopCountAnalysis(self.model, x)
        print('-'*40 + '\n')
        print(f'FLOPs: {flops.total()}, {flops.total() / 1e+9} G, {flops.total() / 1e+6} M' + '\n')
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs) + "\n")
        print('{:<30}  {:<8}'.format('Number of parameters: ', params) + "\n")
        return macs, params, flops.total()
    
    def unit_process_time(self):
        # unit batch size == 1
        data, unit_loader = self._get_data(flag='unit')
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        forward_time = []
        backward_time = []
        total_time = []
        
        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(unit_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)

            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
            inference_time1 = time.time()
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    inference_time2 = time.time()
                        
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                inference_time2 = time.time()
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)

            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()
                    
            inference_time3 = time.time()
            
            unit_forward_time = inference_time2 - inference_time1
            unit_backward_time = inference_time3 - inference_time2
            unit_total_time = inference_time3 - inference_time1
            
            forward_time.append(unit_forward_time)
            backward_time.append(unit_backward_time)
            total_time.append(unit_total_time)
            
            if i == 99:
                break
        
        print(f'The unit forward time is: {(sum(forward_time)/len(forward_time)):.5f} s' + "\n")
        print(f'The unit backward time is: {(sum(backward_time)/len(backward_time)):.5f} s' + "\n")
        print(f'The total time is: {(sum(total_time)/len(total_time)):.5f} s' + "\n")
        
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        
        global done, Score, action_list, rank_list, state, next_state, next_info, min_index, action, reward, last_train_loss, action_train_loss, log
        
        training_time = []
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, save_pt=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()



        action_step = 0
        critic_loss = []
        for epoch in range(self.args.train_epochs):

            action_iter = 0

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                score_list = []
                action_iter += 1

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                action_train_loss += loss.item()
                # compute the layer importance use gradient
                if not self.frame.score_mask.all() == self.frame.done_mask.all():
                    with torch.no_grad():
                        # Encoder
                        for layer in self.model.module.encoder.attn_layers:
                            grad = torch.autograd.grad(loss, layer.attention.parameters(), retain_graph=True)
                            tuples = zip(grad, layer.attention.parameters())
                            importance = list(map(lambda p: (p[0] * p[1]).pow(2).sum(), tuples))
                            score_list.append(sum(importance))

                        # Decoder
                        for layer in self.model.module.decoder.layers:
                            # there are two attention
                            # self_attention
                            grad = torch.autograd.grad(loss, layer.self_attention.parameters(), retain_graph=True)
                            tuples = zip(grad, layer.self_attention.parameters())
                            importance1 = sum(list(map(lambda p: (p[0] * p[1]).pow(2).sum(), tuples)))
                            # cross_attention
                            grad = torch.autograd.grad(loss, layer.cross_attention.parameters(), retain_graph=True)
                            tuples = zip(grad, layer.cross_attention.parameters())
                            importance2 = sum(list(map(lambda p: (p[0] * p[1]).pow(2).sum(), tuples)))

                            score_list.append(sum([importance1, importance2]))

                        score_list = torch.stack(score_list, dim=0).to(DEVICE)
                        Score += score_list


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if action_iter % self.frame.action_interval == 0 and action_iter > 1 and not self.frame.score_mask.all() == self.frame.done_mask.all():

                    if action_step > 0:
                        ll = action_train_loss / self.frame.action_interval
                        if ll < 1.125 * last_train_loss:
                            reward += last_train_loss / ll
                        else:
                            reward += - ll / last_train_loss

                        loss_c, _, _ = self.frame.ac.train_net(state, reward, next_state, action)
                        critic_loss.append(loss_c.item())
                        state = next_state
                        last_train_loss = ll
                        done = judge_done(action_list)
                        action_train_loss *= 0

                        if done:
                            log += 'The {}/{} Attn_layer R is {}, The action is {} \n'.format(min_index + 1,
                                                                                              3,
                                                                                              rank_list, action_list)
                            print('=' * 100 + '\n')
                            print('The {}/{} Attn_layer final R is {}, The action is {} \n'.format(min_index + 1,
                                                                                                     3,
                                                                                                     rank_list,
                                                                                                     action_list))
                            print('=' * 100 + '\n')
                            self.frame.score_mask[min_index] = 1
                            score = Score.masked_fill(self.frame.score_mask == 1, float('inf'))
                            min_index = torch.argmin(score).item()
                            state = next_info[0][min_index]
                            # self.frame.reset_ac(state)
                            action_list = []
                            rank_list = []

                    if not self.frame.score_mask.all() == self.frame.done_mask.all():
                        # print('Start decomposing......')
                        action_step += 1
                        Score /= self.frame.action_interval

                        if action_step == 1:
                            min_index = torch.argmin(Score).item()
                            last_train_loss = action_train_loss / self.frame.action_interval
                            action_train_loss *= 0
                            state = self.frame.arch_info[0][min_index]

                        action = self.frame.ac.choose_action(state)
                        next_info, next_state, real_rank, reward = change(self.model.module, action, min_index, self.frame.enc, self.args.decomp_method)
                        print('-' * 100 + '\n')
                        print('| Layer:{}, Action:{}, Rank:{}|\n'.format(min_index + 1, action, real_rank) )
                        print('-' * 100 + '\n')
                        action_list.append(action)
                        rank_list.append(real_rank)
                        Score *= 0.

            training_time.append(time.time() - epoch_time)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        best_model_path1 = path + '/' + 'checkpoint.pt'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model = torch.load(best_model_path1)
        print(log)
        print(f'The total training time is {sum(training_time):.5f} s\n')
        print(f'Critic loss: {critic_loss}\n')
        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        end_time = time.time()
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print(f'The Testing Time is {(end_time-start_time):.5f} s')
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return


def change_cpcores(model, action, reward, method):
    rank0 = model.W_Q0.shape[1]
    if action < 0:
        return rank0, reward

    rank = real_action(action, method)   # the smallest rank (45, 450)
    # rank = action + 1
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
        except Exception as e:
            reward += -0.005


            model.W_Q0 = nn.Parameter(torch.tensor(Wq0, dtype=torch.float).to(DEVICE))
            model.W_Q1 = nn.Parameter(torch.tensor(Wq1, dtype=torch.float).to(DEVICE))
            model.W_Q2 = nn.Parameter(torch.tensor(Wq2, dtype=torch.float).to(DEVICE))

            model.W_K0 = nn.Parameter(torch.tensor(Wk0, dtype=torch.float).to(DEVICE))
            model.W_K1 = nn.Parameter(torch.tensor(Wk1, dtype=torch.float).to(DEVICE))
            model.W_K2 = nn.Parameter(torch.tensor(Wk2, dtype=torch.float).to(DEVICE))

            model.W_V0 = nn.Parameter(torch.tensor(Wv0, dtype=torch.float).to(DEVICE))
            model.W_V1 = nn.Parameter(torch.tensor(Wv1, dtype=torch.float).to(DEVICE))
            model.W_V2 = nn.Parameter(torch.tensor(Wv2, dtype=torch.float).to(DEVICE))
            
            print(f'Error occurred: {e}')

            return rank0, reward

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


def change(model, action, min_index, enc, method):
    r = 0
    token = []
    real_ranks = []
    
    if method == 'cp':
        change_cores = change_cpcores
    else:
        change_cores = change_svd_cores

    j = 0
    # Encoder
    for layer in model.encoder.attn_layers:
        if j == min_index:
            real_rank, r = change_cores(layer.attention, action, r, method)

        else:
            real_rank, r = change_cores(layer.attention, -4, r, method)

        token += ['MultiHeadAttention-0-8-64-%d' % real_rank]
        real_ranks.append(real_rank)
        j += 1

    # Decoder
    for layer in model.decoder.layers:
        if j == min_index:
            real_rank, r = change_cores(layer.self_attention, action, r, method)
            real_rank, r = change_cores(layer.cross_attention, action, r, method)
            token += ['MultiHeadAttention-1-8-64-%d' % real_rank]
            real_ranks.append(real_rank)
            break

        else:
            real_rank, r = change_cores(layer.self_attention, -4, r, method)
            real_rank, r = change_cores(layer.cross_attention, -4, r, method)
            token += ['MultiHeadAttention-1-8-64-%d' % real_rank]
            real_ranks.append(real_rank)

        j += 1

    next_info, _ = enc(token)
    next_state = next_info[0][min_index]

    return next_info, next_state, real_ranks[min_index], r


def real_action(action, method):
    if method == 'cp':
        # rank = 45 + action * 15
        rank = 45 * (action + 1)
    else:
        # rank = 25 + action * 11
        rank = 25.6 + (action + 1)
        
    return int(rank)


def change_svd_cores(model, action, reward, method):
    rank0 = model.W_Q0.shape[1]
    if action < 0:
        return rank0, reward

    rank = real_action(action, method)   # the smallest rank (25, 256)
    # rank = action + 1
    # input_size = model.W_Q.shape[0]
    if rank0 == rank or rank == 0:
        return rank0, reward

    Wq0 = model.W_Q0.cpu().detach().numpy()
    Wq1 = model.W_Q1.cpu().detach().numpy()

    Wk0 = model.W_K0.cpu().detach().numpy()
    Wk1 = model.W_K1.cpu().detach().numpy()

    Wv0 = model.W_V0.cpu().detach().numpy()
    Wv1 = model.W_V1.cpu().detach().numpy()

    weight = np.array([1.]*rank0)
    wq = tl.cp_to_tensor((weight, [Wq0, Wq1]))
    wk = tl.cp_to_tensor((weight, [Wk0, Wk1]))
    wv = tl.cp_to_tensor((weight, [Wv0, Wv1]))
    Wq = tl.tensor(wq)
    Wk = tl.tensor(wk)
    Wv = tl.tensor(wv)
    
    f = lambda x: torch.tensor([torch.isnan(a).any() for a in x])
    while True:
        try:
            _, WQ = parafac(Wq, rank)
            _, WK = parafac(Wk, rank)
            _, WV = parafac(Wv, rank)
            # check whether WQ, WK, WV contains nan
            for item in [WQ, WK, WV]:
                assert f(item).any() == False
                
            break
        except:
            reward += -0.005


            model.W_Q0 = nn.Parameter(torch.tensor(Wq0, dtype=torch.float).to(DEVICE))
            model.W_Q1 = nn.Parameter(torch.tensor(Wq1, dtype=torch.float).to(DEVICE))

            model.W_K0 = nn.Parameter(torch.tensor(Wk0, dtype=torch.float).to(DEVICE))
            model.W_K1 = nn.Parameter(torch.tensor(Wk1, dtype=torch.float).to(DEVICE))

            model.W_V0 = nn.Parameter(torch.tensor(Wv0, dtype=torch.float).to(DEVICE))
            model.W_V1 = nn.Parameter(torch.tensor(Wv1, dtype=torch.float).to(DEVICE))

            return rank0, reward

    model.W_Q0 = nn.Parameter(torch.tensor(WQ[0], dtype=torch.float).to(DEVICE))
    model.W_Q1 = nn.Parameter(torch.tensor(WQ[1], dtype=torch.float).to(DEVICE))

    model.W_K0 = nn.Parameter(torch.tensor(WK[0], dtype=torch.float).to(DEVICE))
    model.W_K1 = nn.Parameter(torch.tensor(WK[1], dtype=torch.float).to(DEVICE))

    model.W_V0 = nn.Parameter(torch.tensor(WV[0], dtype=torch.float).to(DEVICE))
    model.W_V1 = nn.Parameter(torch.tensor(WV[1], dtype=torch.float).to(DEVICE))

    return rank, reward
