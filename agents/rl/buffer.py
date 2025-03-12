import numpy as np
import torch

class ReplayBuffer:
    """
    经验回放缓冲区，用于存储和采样强化学习代理的交互经验。
    专为PPO算法设计，支持存储状态、动作、奖励、值、动作概率和终止信号。
    """
    
    def __init__(self, state_dim, action_dim, buffer_size=1000, device='cpu'):
        """
        初始化经验回放缓冲区。
        
        参数:
            state_dim (int): 状态向量的维度
            action_dim (int): 动作向量的维度
            buffer_size (int): 缓冲区大小
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # 初始化缓冲区
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        self.ptr = 0  # 指向下一个要填充的位置
        self.size = 0  # 当前缓冲区中的样本数量
        
    def add(self, state, action, reward, value, log_prob, done):
        """
        向缓冲区添加一个经验样本。
        
        参数:
            state: 状态向量
            action: 动作向量
            reward (float): 奖励值
            value (float): 值函数估计
            log_prob (float): 动作的对数概率
            done (bool): 是否是终止状态
        """
        # 确保输入是张量
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        # 存储经验
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_batch(self, batch_size=None):
        """
        从缓冲区获取一个批次的样本。
        
        参数:
            batch_size (int, optional): 批次大小，如果为None则返回所有样本
            
        返回:
            dict: 包含批次数据的字典
        """
        if batch_size is None:
            batch_size = self.size
        
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'dones': self.dones[indices]
        }
    
    def get_all(self):
        """
        获取缓冲区中的所有样本。
        
        返回:
            dict: 包含所有样本的字典
        """
        return {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'values': self.values[:self.size],
            'log_probs': self.log_probs[:self.size],
            'dones': self.dones[:self.size]
        }
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0
    
    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        计算广义优势估计(GAE)和折扣回报。
        
        参数:
            last_value (float): 最后一个状态的值函数估计
            gamma (float): 折扣因子
            gae_lambda (float): GAE lambda参数
            
        返回:
            tuple: (returns, advantages)
        """
        # 获取所有样本
        rewards = self.rewards[:self.size].clone()
        values = self.values[:self.size].clone()
        dones = self.dones[:self.size].clone()
        
        # 初始化回报和优势
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # 初始化下一个值和下一个优势
        next_value = last_value
        next_advantage = 0
        
        # 从后向前计算
        for t in reversed(range(self.size)):
            # 如果是终止状态，下一个值为0
            if dones[t]:
                next_value = 0
                next_advantage = 0
            
            # 计算TD误差
            delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
            
            # 计算GAE
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t].float()) * next_advantage
            
            # 计算回报
            returns[t] = advantages[t] + values[t]
            
            # 更新下一个值和优势
            next_value = values[t]
            next_advantage = advantages[t]
        
        return returns, advantages
