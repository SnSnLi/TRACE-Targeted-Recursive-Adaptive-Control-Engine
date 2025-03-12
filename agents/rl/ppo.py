import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from ...meta_controller import MetaController  # 导入 MetaController
from .buffer import ReplayBuffer  # 导入 ReplayBuffer

class PPOTrainer:
    """
    优化后的 PPO 算法实现，用于训练元控制器。
    包含批次更新、GAE 优势估计和梯度裁剪。
    """
    
    def __init__(self, meta_controller: MetaController, lr=3e-4, gamma=0.99, epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, update_epochs=4,
                 batch_size=64):
        """
        初始化 PPO 训练器。
        
        Args:
            meta_controller (MetaController): 元控制器实例
            lr (float): 学习率
            gamma (float): 折扣因子
            epsilon (float): PPO 裁剪参数
            value_coef (float): 值函数损失权重
            entropy_coef (float): 熵正则化权重
            max_grad_norm (float): 梯度裁剪最大范数
            update_epochs (int): 每批数据更新轮数
            batch_size (int): 批次大小
        """
        self.meta_controller = meta_controller.to(next(meta_controller.parameters()).device)
        self.optimizer = optim.Adam(meta_controller.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = next(meta_controller.parameters()).device

    def evaluate(self, query_embs, result_embs, uncertainties, actions=None):
        """
        评估元控制器输出。
        
        Args:
            query_embs (torch.Tensor): 查询嵌入
            result_embs (torch.Tensor): 结果嵌入
            uncertainties (torch.Tensor): 不确定性
            actions (torch.Tensor, optional): 动作，用于计算 log_prob
            
        Returns:
            tuple: (action_probs, values, entropy)
        """
        action_probs, values = self.meta_controller(query_embs, result_embs, uncertainties)
        dist = Categorical(action_probs)
        entropy = dist.entropy().mean()
        log_probs = dist.log_prob(actions) if actions is not None else None
        return action_probs, values.squeeze(-1), entropy, log_probs

    def update(self, buffer: ReplayBuffer, last_value=0):
        """
        使用 PPO 更新元控制器。
        
        Args:
            buffer (ReplayBuffer): 经验回放缓冲区
            last_value (float): 最后一个状态的值函数估计
            
        Returns:
            dict: 训练统计信息
        """
        # 计算回报和 GAE 优势
        returns, advantages = buffer.compute_returns(last_value, self.gamma, gae_lambda=0.95)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化优势

        # 获取所有样本
        batch = buffer.get_all()
        query_embs = batch['states'][:, :, :self.meta_controller.embedding_dim]
        result_embs = batch['states'][:, :, self.meta_controller.embedding_dim:2*self.meta_controller.embedding_dim]
        uncertainties = batch['states'][:, :, -1].unsqueeze(-1)
        actions = batch['actions'].squeeze(-1)
        old_log_probs = batch['log_probs']
        old_values = batch['values']
        returns = returns
        advantages = advantages

        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.update_epochs):
            indices = torch.randperm(buffer.size)
            for start in range(0, buffer.size, self.batch_size):
                end = min(start + self.batch_size, buffer.size)
                batch_indices = indices[start:end]

                # 获取批次数据
                batch_query_embs = query_embs[batch_indices]
                batch_result_embs = result_embs[batch_indices]
                batch_uncertainties = uncertainties[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 评估当前策略
                _, batch_values, batch_entropy, batch_new_log_probs = self.evaluate(
                    batch_query_embs, batch_result_embs, batch_uncertainties, batch_actions
                )

                # 计算比率
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算值函数损失
                value_loss = nn.MSELoss()(batch_values, batch_returns)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * batch_entropy

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta_controller.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += batch_entropy.item()
                num_updates += 1

        # 计算平均损失
        avg_stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        return avg_stats

    def save(self, path):
        """保存模型和优化器状态"""
        torch.save({
            'meta_controller_state': self.meta_controller.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型和优化器状态"""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_controller.load_state_dict(checkpoint['meta_controller_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
