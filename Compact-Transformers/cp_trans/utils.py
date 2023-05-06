import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .attention import DEVICE
import tensorly as tl
from tensorly.decomposition import parafac
import pdb
# from numpy import seterr
# seterr(all='raise')

# fix the random seed
def set_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed, details are ",e)

    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)

# check the actor converge
def judge_done(rank_list, num_same=None):
    least_len = num_same if num_same is not None else 5

    if len(rank_list) < least_len:
        done = False
        return done
    if len(rank_list) >= 5:
        done = True
        return done

    num = 4 if num_same is None else num_same-1
    same = 1
    for i in range(num):
        if rank_list[-1] == rank_list[-2-i]:
            same += 1
        else:
            done = False
            return done
    done = True
    return done

# action -> rank
def real_action(theta, action, start_rank, end_rank, gap):
    assert (start_rank - end_rank) % gap == 0

    if theta > 0.:
        if action == 0:
            return action
        else:
            return start_rank + (action - 1) * gap
    else:
        return start_rank + action * gap


def change_cpcores(theta, model, action, reward, start_rank, end_rank, gap):

    rank = real_action(theta, action, start_rank, end_rank, gap)
    f = lambda x: torch.tensor([torch.isnan(torch.from_numpy(a)).any() for a in x])

    if model.is_decomposed:
        origin_rank = model.W_Q0.shape[1]

        if origin_rank == rank or action < 0:
            return origin_rank, reward

        Wq0 = (model.W_Q0).clone().cpu().detach().numpy()
        Wq1 = (model.W_Q1).clone().cpu().detach().numpy()
        Wq2 = (model.W_Q2).clone().cpu().detach().numpy()

        Wk0 = (model.W_K0).clone().cpu().detach().numpy()
        Wk1 = (model.W_K1).clone().cpu().detach().numpy()
        Wk2 = (model.W_K2).clone().cpu().detach().numpy()

        Wv0 = (model.W_V0).clone().cpu().detach().numpy()
        Wv1 = (model.W_V1).clone().cpu().detach().numpy()
        Wv2 = (model.W_V2).clone().cpu().detach().numpy()

        weight = np.array([1.]*origin_rank)
        Wq = tl.tensor(tl.cp_to_tensor((weight, [Wq0, Wq1, Wq2])))
        Wk = tl.tensor(tl.cp_to_tensor((weight, [Wk0, Wk1, Wk2])))
        Wv = tl.tensor(tl.cp_to_tensor((weight, [Wv0, Wv1, Wv2])))

        # rank==0 means, the original state of attention
        if rank == 0:
            weight_q = nn.Parameter(torch.tensor(Wq, dtype=torch.float).reshape(model.dim, -1)).cuda()
            weight_k = nn.Parameter(torch.tensor(Wk, dtype=torch.float).reshape(model.dim, -1)).cuda()
            weight_v = nn.Parameter(torch.tensor(Wv, dtype=torch.float).reshape(model.dim, -1)).cuda()

            model.q = nn.Linear(model.dim, model.dim, bias=False).cuda()
            model.k = nn.Linear(model.dim, model.dim, bias=False).cuda()
            model.v = nn.Linear(model.dim, model.dim, bias=False).cuda()

            model.q.weight = weight_q
            model.k.weight = weight_k
            model.v.weight = weight_v

            model.move_cpcores()
            return rank, reward
    else:
        origin_rank = 0
        if action < 0 or rank == 0:
            return origin_rank, reward

        Wq = tl.tensor((model.q.weight).clone().cpu().detach().numpy().reshape(model.dim, model.num_heads, -1))
        Wk = tl.tensor((model.k.weight).clone().cpu().detach().numpy().reshape(model.dim, model.num_heads, -1))
        Wv = tl.tensor((model.v.weight).clone().cpu().detach().numpy().reshape(model.dim, model.num_heads, -1))

    try:
        # pdb.trace()
        _, WQ = parafac(Wq, rank)
        _, WK = parafac(Wk, rank)
        _, WV = parafac(Wv, rank)

        # check whether WQ, WK, WV contains nan
        for item in [WQ, WK, WV]:
            # print(f(item).any())
            assert f(item).any() == False
            # Replace oversized elements (typical 1e+9 or 1e+14 since divided by 0) with 100
            for i in range(len(item)):
                alter = np.full(item[i].shape, 100)
                item[i] = np.where(np.abs(item[i]) > 100, alter, item[i])

        print(f'The action is {action}, the real rank is {rank}')
    except:
        reward += -2.5
        print(f'The action is {action}, the real rank is {origin_rank}')
        return origin_rank, reward

    model.W_Q0 = nn.Parameter(torch.tensor(WQ[0], dtype=torch.float).cuda())
    model.W_Q1 = nn.Parameter(torch.tensor(WQ[1], dtype=torch.float).cuda())
    model.W_Q2 = nn.Parameter(torch.tensor(WQ[2], dtype=torch.float).cuda())

    model.W_K0 = nn.Parameter(torch.tensor(WK[0], dtype=torch.float).cuda())
    model.W_K1 = nn.Parameter(torch.tensor(WK[1], dtype=torch.float).cuda())
    model.W_K2 = nn.Parameter(torch.tensor(WK[2], dtype=torch.float).cuda())

    model.W_V0 = nn.Parameter(torch.tensor(WV[0], dtype=torch.float).cuda())
    model.W_V1 = nn.Parameter(torch.tensor(WV[1], dtype=torch.float).cuda())
    model.W_V2 = nn.Parameter(torch.tensor(WV[2], dtype=torch.float).cuda())
    try:
        model.move_qkv()
    except:
        pass

    return rank, reward


def change_svd_cores(theta, model, action, reward, start_rank, end_rank, gap):
    rank = real_action(theta, action, start_rank, end_rank, gap)
    f = lambda x: torch.tensor([torch.isnan(torch.from_numpy(a)).any() for a in x])

    if model.is_decomposed:
        origin_rank = model.W_Q0.shape[1]

        if origin_rank == rank or action < 0:
            return origin_rank, reward

        Wq0 = (model.W_Q0).clone().cpu().detach().numpy()
        Wq1 = (model.W_Q1).clone().cpu().detach().numpy()

        Wk0 = (model.W_K0).clone().cpu().detach().numpy()
        Wk1 = (model.W_K1).clone().cpu().detach().numpy()

        Wv0 = (model.W_V0).clone().cpu().detach().numpy()
        Wv1 = (model.W_V1).clone().cpu().detach().numpy()


        weight = np.array([1.] * origin_rank)
        Wq = tl.tensor(tl.cp_to_tensor((weight, [Wq0, Wq1])))
        Wk = tl.tensor(tl.cp_to_tensor((weight, [Wk0, Wk1])))
        Wv = tl.tensor(tl.cp_to_tensor((weight, [Wv0, Wv1])))

        # rank==0 means, the original state of attention
        if rank == 0:
            weight_q = nn.Parameter(torch.tensor(Wq, dtype=torch.float).reshape(model.dim, -1)).cuda()
            weight_k = nn.Parameter(torch.tensor(Wk, dtype=torch.float).reshape(model.dim, -1)).cuda()
            weight_v = nn.Parameter(torch.tensor(Wv, dtype=torch.float).reshape(model.dim, -1)).cuda()

            model.q = nn.Linear(model.dim, model.dim, bias=False).cuda()
            model.k = nn.Linear(model.dim, model.dim, bias=False).cuda()
            model.v = nn.Linear(model.dim, model.dim, bias=False).cuda()

            model.q.weight = weight_q
            model.k.weight = weight_k
            model.v.weight = weight_v

            model.move_cpcores()
            return rank, reward
    else:
        origin_rank = 0
        if action < 0 or rank == 0:
            return origin_rank, reward

        Wq = tl.tensor((model.q.weight).clone().cpu().detach().numpy())
        Wk = tl.tensor((model.k.weight).clone().cpu().detach().numpy())
        Wv = tl.tensor((model.v.weight).clone().cpu().detach().numpy())

    try:
        # pdb.trace()
        _, WQ = parafac(Wq, rank)
        _, WK = parafac(Wk, rank)
        _, WV = parafac(Wv, rank)
        # check whether WQ, WK, WV contains nan
        for item in [WQ, WK, WV]:
            # print(f(item).any())
            assert f(item).any() == False
            # Replace oversized elements (typical 1e+9 or 1e+14 since divided by 0) with 100
            for i in range(len(item)):
                alter = np.full(item[i].shape, 100)
                item[i] = np.where(np.abs(item[i]) > 100, alter, item[i])

        print(f'The action is {action}, the real rank is {rank}')
    except:
        reward += -2.5
        print(f'The action is {action}, the real rank is {origin_rank}')
        return origin_rank, reward

    model.W_Q0 = nn.Parameter(torch.tensor(WQ[0], dtype=torch.float).cuda())
    model.W_Q1 = nn.Parameter(torch.tensor(WQ[1], dtype=torch.float).cuda())

    model.W_K0 = nn.Parameter(torch.tensor(WK[0], dtype=torch.float).cuda())
    model.W_K1 = nn.Parameter(torch.tensor(WK[1], dtype=torch.float).cuda())

    model.W_V0 = nn.Parameter(torch.tensor(WV[0], dtype=torch.float).cuda())
    model.W_V1 = nn.Parameter(torch.tensor(WV[1], dtype=torch.float).cuda())
    try:
        model.move_qkv()
    except:
        pass

    return rank, reward

# this is an example for cp or svd decomposition with layer importance
def step_follow_cp(model, encoder, action, last_averAcc, min_index, method, start_rank, end_rank, gap):
    reward = 0
    token = []
    real_ranks = []

    if method == 'cp':
        change_cores = change_cpcores
    else:
        change_cores = change_svd_cores

    j = 0
    for block in model.model.module.TransformerEncoder.layers:
        if j == min_index:
            real_rank, reward = change_cores(block.attn, action, reward, start_rank, end_rank, gap)
        else:
            real_rank, reward = change_cores(block.attn, -1, reward, start_rank, end_rank, gap)
        token += ['MultiHeadAttention-0-5-2-%d' % real_rank]
        real_ranks.append(real_rank)
        j += 1

    next_info, _ = encoder(token)
    bestAcc, averAcc, Y_true, Y_pred, Score = model.train()
    next_state = next_info[0][min_index]
    # train_acc = train_accs[-1]
    # acc = accs[-1]
    if last_averAcc - averAcc <= 0.1:
        reward += averAcc / last_averAcc
    else:
        reward += - last_averAcc / averAcc

    return next_info, next_state, bestAcc, averAcc, real_ranks[min_index], reward, Y_true, Y_pred, Score

# this is an example for cp decomposition without layer importance, namely uniform
def step_follow_uniform(model, encoder, action, last_averAcc, min_index, method, start_rank, end_rank, gap):
    reward = 0
    token = []
    real_ranks = []

    if method == 'cp':
        change_cores = change_cpcores
    else:
        change_cores = change_svd_cores

    for block in model.model.module.TransformerEncoder.layers:
        real_rank, reward = change_cores(block.attn, action, reward, start_rank, end_rank, gap)
        token += ['MultiHeadAttention-0-5-2-%d' % real_rank]
        real_ranks.append(real_rank)

    next_info, _ = encoder(token)
    bestAcc, averAcc, Y_true, Y_pred, Score = model.train()
    next_state = next_info[0][min_index]
    # train_acc = train_accs[-1]
    # acc = accs[-1]
    if last_averAcc - averAcc <= 0.1:
        reward += averAcc / last_averAcc
    else:
        reward += - last_averAcc / averAcc

    return next_info, next_state, bestAcc, averAcc, real_ranks[min_index], reward, Y_true, Y_pred, Score

