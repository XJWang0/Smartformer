import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import DEVICE

GAMMA = 0.95  # discount
LR = 0.01

class ACnet(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim):
        super(ACnet, self).__init__()
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, hidden_dim)
        self.pi2 = nn.Linear(hidden_dim, a_dim)
        self.v1 = nn.Linear(s_dim, hidden_dim)
        self.v2 = nn.Linear(hidden_dim, 1)
        self.set_init([self.pi1, self.pi2, self.v1, self.v2])

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        logits = self.pi2(pi1)

        v1 = F.relu(self.v1(x))
        values = self.v2(v1)

        return logits, values

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

class AC(object):
    def __init__(self, state, a_dim):
        self.state_dim = state.shape[-1]
        self.action_dim = a_dim
        self.net = ACnet(self.state_dim, 40, self.action_dim).to(DEVICE)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.distribution = torch.distributions.Categorical
        self.I = 1.

    def choose_action(self, observation):
        observation = observation.float().to(DEVICE)
        logits, _ = self.net.forward(observation)
        prob = F.softmax(logits, dim=-1)
        #m = self.distribution(prob)
        action = prob.argmax(dim=-1).item()

        return action

    def train_net(self, state, reward, next_state, action):
        s, s_ = state.float().to(DEVICE), next_state.float().to(DEVICE)
        logits, values = self.net.forward(s)  # v(s)
        _, v_ = self.net.forward(s_)  # v(s')

        # loss_c = self.loss_func(reward + GAMMA * v_, v)
        td_error = reward + GAMMA * v_ - values
        loss_c = self.I * td_error * values

        # with torch.no_grad():
            # td_error = reward + GAMMA * v_ - v

        probs = F.softmax(logits, dim=-1)
        m = self.distribution(probs)
        a = torch.tensor(action).to(DEVICE)
        exp_v = self.I * td_error * m.log_prob(a)
        loss_a = -exp_v
        total_loss = (loss_a + loss_c).mean()
        self.I = self.I * GAMMA

        self.optim.zero_grad()
        total_loss.backward(retain_graph = True)
        self.optim.step()



    def change_out(self, R):
        new_weight = torch.randn(self.action_dim - R, 40)
        new_bias = torch.randn(self.action_dim - R)
        self.net.pi2.weight = nn.Parameter(torch.tensor(new_weight).to(DEVICE))
        self.net.pi2.bias = nn.Parameter(torch.tensor(new_bias).to(DEVICE))
