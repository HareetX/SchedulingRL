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

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
