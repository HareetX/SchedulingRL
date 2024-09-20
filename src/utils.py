import numpy as np
import torch

def compute_advantage(td_delta, gamma, lmbda):
    td_delta = td_delta.detach().numpy()
    discount_factor = gamma * lmbda
    advantages = np.zeros_like(td_delta)
    advantage = 0
    for t in reversed(range(len(td_delta))):
        advantage = td_delta[t] + discount_factor * advantage
        advantages[t] = advantage
    return torch.tensor(advantages, dtype=torch.float)
