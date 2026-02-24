import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# Actor with attention-based input weighting
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, state_size),
            nn.Softmax(dim=-1)
        )
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.attn(x)  # Compute attention weights
        return self.net(x * w)  # Weighted state → action


# Centralized critic for all agents
class Critic(nn.Module):
    def __init__(self, state_size, action_size, n_agents, hidden=128):
        super().__init__()
        self.n_agents = n_agents
        self.encoder = nn.Sequential(
            nn.Linear(state_size + action_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.q_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, states, actions):
        B = states.shape[0]
        s = states.view(B, self.n_agents, -1)
        a = actions.view(B, self.n_agents, -1)
        sa = torch.cat([s, a], dim=-1)
        h = self.encoder(sa).sum(dim=1)  # Aggregate across agents
        return self.q_net(h)


# Multi-Agent DDPG (MADDPG)
class MADDPG:
    def __init__(self, state_size, action_size, n_agents):
        self.s_dim, self.a_dim, self.n = state_size, action_size, n_agents
        self.gamma = 0.9
        self.tau = 0.005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actors = [Actor(state_size, action_size).to(self.device) for _ in range(n_agents)]
        self.actors_t = [Actor(state_size, action_size).to(self.device) for _ in range(n_agents)]
        self.opt_a = [optim.Adam(self.actors[i].parameters(), lr=1e-4, weight_decay=1e-5) for i in range(n_agents)]

        self.critic = Critic(state_size, action_size, n_agents).to(self.device)
        self.critic_t = Critic(state_size, action_size, n_agents).to(self.device)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-5)

        # Initialize target networks
        for i in range(n_agents):
            self.hard_update(self.actors[i], self.actors_t[i])
        self.hard_update(self.critic, self.critic_t)

        self.mem = deque(maxlen=1_000_000)
        self.batch = 128
        self.noise_scale = 0.1

    def hard_update(self, src, dst):
        for sp, dp in zip(src.parameters(), dst.parameters()):
            dp.data.copy_(sp.data)

    def soft_update(self, src, dst):
        for sp, dp in zip(src.parameters(), dst.parameters()):
            dp.data.mul_(1 - self.tau).add_(sp.data, alpha=self.tau)

    def act(self, states, noise_scale=None):
        if noise_scale is None:
            noise_scale = self.noise_scale
        acts = []
        for i, s in enumerate(states):
            s = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            with torch.no_grad():
                a = self.actors[i](s).cpu().numpy()[0]
            a += noise_scale * np.random.randn(self.a_dim)
            acts.append(np.clip(a, 0, 1))
        return acts

    def remember(self, s, a, r, s2, d):
        self.mem.append((s, a, r, s2, d))

    def learn(self):
        if len(self.mem) < self.batch:
            return
        batch = random.sample(self.mem, self.batch)
        s, a, r, s2, d = map(lambda x: torch.FloatTensor(np.array(x)).to(self.device), zip(*batch))

        s = s.view(self.batch, -1)
        a = a.view(self.batch, -1)
        s2 = s2.view(self.batch, -1)

        # Critic update
        with torch.no_grad():
            a2 = torch.cat([self.actors_t[i](s2[:, i * self.s_dim:(i + 1) * self.s_dim]) for i in range(self.n)], 1)
            q_tgt = r.unsqueeze(1) + self.gamma * self.critic_t(s2, a2) * (1 - d.unsqueeze(1))
        q = self.critic(s, a)
        loss_c = F.mse_loss(q, q_tgt)
        self.opt_c.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.opt_c.step()

        # Actor updates
        for i in range(self.n):
            a_pred = a.clone()
            a_pred[:, i * self.a_dim:(i + 1) * self.a_dim] = \
                self.actors[i](s[:, i * self.s_dim:(i + 1) * self.s_dim])
            loss_a = -self.critic(s, a_pred).mean()
            self.opt_a[i].zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=0.5)
            self.opt_a[i].step()
            self.soft_update(self.actors[i], self.actors_t[i])

        self.soft_update(self.critic, self.critic_t)

    def save(self, name, episode=None, hist=None, eps=None):
        os.makedirs("models", exist_ok=True)
        path = os.path.join("models", name)
        ckpt = {f"actor_{i}": self.actors[i].state_dict() for i in range(self.n)}
        ckpt.update({f"actor_{i}_opt": self.opt_a[i].state_dict() for i in range(self.n)})
        ckpt.update({f"actor_t_{i}": self.actors_t[i].state_dict() for i in range(self.n)})
        ckpt["critic"] = self.critic.state_dict()
        ckpt["critic_t"] = self.critic_t.state_dict()
        ckpt["critic_opt"] = self.opt_c.state_dict()
        ckpt.update({
            "state_size": self.s_dim, "action_size": self.a_dim,
            "num_agents": self.n, "gamma": self.gamma, "tau": self.tau,
            "episode": episode, "reward_history": hist, "epsilon": eps
        })
        torch.save(ckpt, path)
        print(f"Saved to {path}")
        return path

    def load(self, filename, load_opt=True):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found")
        ckpt = torch.load(filename, map_location=self.device)
        for i in range(self.n):
            self.actors[i].load_state_dict(ckpt[f"actor_{i}"])
            self.actors_t[i].load_state_dict(ckpt[f"actor_t_{i}"])
            if load_opt:
                self.opt_a[i].load_state_dict(ckpt[f"actor_{i}_opt"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_t.load_state_dict(ckpt["critic_t"])
        if load_opt:
            self.opt_c.load_state_dict(ckpt["critic_opt"])
        print("Model loaded")
        return {k: ckpt.get(k) for k in ("episode", "reward_history", "epsilon")}
