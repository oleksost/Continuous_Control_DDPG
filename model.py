"""
Actor-Critic definitions.
"""

import torch
import torch.nn as nn
import numpy as np


def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)


"""
    Actor
"""


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0.0,nodes=[128, 128]):
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, nodes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(nodes[0]),
            nn.Linear(nodes[0], nodes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(nodes[1]),
            nn.Linear(nodes[1], action_size),
            nn.Tanh()
        )

        self.model.apply(self.init_weights)

    def forward(self, state):
        # return torch.clamp(self.model(state), -1.0, 1.0)
        return self.model(state)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # m.weight.data= fanin_init(m.weight.data.size())
            m.bias.data.fill_(0.1)


"""
    Critic
"""


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=0.0, nodes=[128, 128]):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.model_input = nn.Sequential(
            nn.Linear(state_size, nodes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(nodes[0]),

        )
        self.model_output = nn.Sequential(
            nn.Linear(nodes[0] + action_size, nodes[1]),
            nn.ReLU(),
            nn.Linear(nodes[1], 1),
        )

        self.model_input.apply(self.init_weights)
        self.model_output.apply(self.init_weights)

    def forward(self, state, action):
        i = torch.cat([self.model_input(state), action], dim=1)
        return self.model_output(i)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # m.weight.data= fanin_init(m.weight.data.size())
            m.bias.data.fill_(0.1)
