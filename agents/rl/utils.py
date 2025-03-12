import random
import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.cat(state),
            torch.tensor(action),
            torch.tensor(reward),
            torch.cat(next_state),
            torch.tensor(done)
        )
        
    def __len__(self):
        return len(self.buffer)
        
def compute_rewards(retrieval_results, ground_truth, metrics=["hit@1", "recall"]):
    """
    Calculate rewards based on retrieval performance
    """
    rewards = 0
    
    if "hit@1" in metrics:
        hit_1 = 1 if retrieval_results[0] in ground_truth else 0
        rewards += hit_1
        
    if "recall" in metrics:
        recall = len(set(retrieval_results) & set(ground_truth)) / len(ground_truth)
        rewards += recall
        
    return rewards