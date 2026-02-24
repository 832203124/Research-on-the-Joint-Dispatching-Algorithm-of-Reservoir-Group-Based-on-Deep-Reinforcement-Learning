import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, n_agents, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * (state_size + action_size), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], 1))
