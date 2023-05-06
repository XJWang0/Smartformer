import torch
from cp_trans import DEVICE, EncoderNet, AC, change_cpcores, change_svd_cores


class RL_Framework():
    def __init__(self, args):
        # AC and decomposition
        self.num_layers = args.cp_layers
        self.action_interval = 5
        self.log = ''

        self.Score = torch.tensor([0.] * self.num_layers).cuda()
        self.done_mask = torch.tensor([1.] * self.num_layers).cuda()
        self.args = args

        self.num_heads, self.start_rank, self.end_rank, self.gap, action_dim = self.get_rank()

        rl_list = ['smart', 'svd', 'uniform']
        if self.args.model_type in rl_list:
            token_list = ['MultiHeadAttention-0-%d-64-%d' % (self.num_heads, self.args.cp_rank)] * self.num_layers

            self.enc = EncoderNet(512, 64)
            self.arch_info = self.get_state(token_list)

            if self.args.model_type == 'uniform':
                self.score_mask = torch.tensor([1.] * self.num_layers).cuda()
                S = self.arch_info.reshape(-1)
            else:
                self.score_mask = torch.tensor([0.] * self.num_layers).cuda()
                S = self.arch_info[0][0]

            self.ac = AC(S, action_dim)
        else:
            self.score_mask = torch.tensor([1.] * self.num_layers).cuda()

        if args.not_decomposed_layers is not None:
            # Choose the layers not to be decomposed
            for i in args.not_decomposed_layers:
                self.score_mask[int(i)] = 1.
                self.Score[int(i)] = float('inf')

    def get_state(self, token):
        out, _ = self.enc(token)
        return out

    def choose_action(self, state):
        return self.ac.choose_action(state)

    def train(self, state, reward, next_state, action):
        self.ac.train_net(state, reward, next_state, action)

    def get_rank(self):
        if self.args.theta > 0.:
            if self.num_layers == 6 or self.num_layers == 7:
                if self.args.model_type == 'svd':
                    return 4, 12, 128, 4, 31
                else:
                    return 4, 20, 200, 9, 22
            elif self.num_layers == 4 or self.num_layers == 2:
                if self.args.model_type == 'svd':
                    return 2, 6, 64, 2, 31
                else:
                    return 2, 8, 85, 7, 13
        else:
            if self.num_layers == 6 or self.num_layers == 7:
                if self.args.model_type == 'svd':
                    return 4, 12, 128, 4, 30
                else:
                    return 4, 20, 200, 9, 21
            elif self.num_layers == 4 or self.num_layers == 2:
                if self.args.model_type == 'svd':
                    return 2, 6, 64, 2, 30
                else:
                    return 2, 8, 85, 7, 12



# compute layer importance
def compute_layer_importance(loss, blocks, ):
    score_list = []
    grad_list = []

    with torch.no_grad():
        # Encoder
        for layer in blocks:
            grad = torch.autograd.grad(loss, layer.self_attn.parameters(), retain_graph=True, create_graph=False)
            tuples = zip(grad, layer.self_attn.parameters())
            importance = list(map(lambda p: (p[0] * p[1]).pow(2).sum(), tuples))
            grad_ = [x.sum() for x in grad]
            score_list.append(sum(importance))
            grad_list.append(sum(grad_))

        score_list = torch.stack(score_list, dim=0).cuda()
        grad_list = torch.stack(grad_list, dim=0).cuda()
    return score_list, grad_list


def step(theta, model, action, min_index, rl_framework):
    r = 0
    token = []
    real_ranks = []
    num_head, start_rank, end_rank, gap = rl_framework.num_heads, rl_framework.start_rank, rl_framework.end_rank, rl_framework.gap
    # pdb.set_trace()

    if rl_framework.args.model_type == 'svd':
        change_cores = change_svd_cores
    else:
        change_cores = change_cpcores

    j = 0
    # Encoder
    for layer in model.classifier.blocks:
        if rl_framework.args.model_type == 'uniform':
            real_rank, r = change_cores(theta, layer.self_attn, action, r, start_rank, end_rank, gap)
        else:
            if j == min_index:
                real_rank, r = change_cores(theta, layer.self_attn, action, r, start_rank, end_rank, gap)
            else:
                real_rank, r = change_cores(theta, layer.self_attn, -100, r, start_rank, end_rank, gap)

        token += ['MultiHeadAttention-0-%d-64-%d' % (num_head, real_rank)]
        real_ranks.append(real_rank)
        j += 1

    next_info = rl_framework.get_state(token)


    if rl_framework.args.model_type == 'uniform':
        return_ranks = real_ranks
        next_state = next_info.reshape(-1)
    else:
        return_ranks = real_ranks[min_index]
        next_state = next_info[0][min_index]

    return next_info, next_state, return_ranks, r