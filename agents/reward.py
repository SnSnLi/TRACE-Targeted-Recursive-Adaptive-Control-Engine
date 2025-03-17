import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HierarchicalRewardFunction(nn.Module):
    def __init__(self, 
                 quality_weight=1.0, 
                 uncertainty_weight=0.5, 
                 step_penalty=0.1,
                 early_stop_bonus=0.5,
                 late_stop_penalty=0.3,
                 ortho_weight=1.0):
        """
        初始化分层奖励函数
        
        参数:
            quality_weight (float): 结果质量相关奖励的基础权重
            uncertainty_weight (float): 不确定性奖励的基础权重
            step_penalty (float): 每步的惩罚（鼓励尽快完成任务）
            early_stop_bonus (float): 提前停止且结果好的奖励
            late_stop_penalty (float): 过晚停止的惩罚
            ortho_weight (float): 正交性惩罚的权重，用于降低不同奖励之间的相关性
        """
        super(HierarchicalRewardFunction, self).__init__()
        self.quality_weight = quality_weight
        self.uncertainty_weight = uncertainty_weight
        self.step_penalty = step_penalty
        self.early_stop_bonus = early_stop_bonus
        self.late_stop_penalty = late_stop_penalty
        self.ortho_weight = ortho_weight
        
        # 动态权重，作为可学习参数
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 短期奖励权重
        self.beta = nn.Parameter(torch.tensor(1.0))   # 中期奖励权重
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 长期奖励权重

    def compute_short_term_reward(self, current_quality, previous_quality):
        """
        短期奖励：直接的质量改进
        
        参数:
            current_quality (Tensor): 当前检索结果质量 (batch,)
            previous_quality (Tensor): 上一步检索结果质量 (batch,)
            
        返回:
            Tensor: 短期奖励 (batch,)
        """
        return current_quality - previous_quality

    def compute_mid_term_reward(self, uncertainty, action, step_count, max_steps):
        """
        中期奖励：基于不确定性奖励和步数惩罚
        
        参数:
            uncertainty (float): 当前不确定性 (标量, 已归一化到 [0,1])
            action (int): 采取的动作 (0: 停止, 1: 继续)
            step_count (float): 当前步数
            max_steps (float): 最大步数
            
        返回:
            float: 中期奖励
        """
        reward = 0.0
        if action == 0:
            # 当采取停止动作时
            if uncertainty < 0.3:
                reward += self.early_stop_bonus
            elif uncertainty > 0.7:
                reward -= self.early_stop_bonus
        else:
            # 当采取继续动作时
            if uncertainty > 0.5:
                reward += self.uncertainty_weight * uncertainty
            else:
                reward -= self.uncertainty_weight * (1 - uncertainty)
            if step_count > 0.8 * max_steps:
                reward -= self.late_stop_penalty
        reward += -self.step_penalty * (step_count / max_steps)
        return reward

    def compute_long_term_reward(self, final_quality, step_count, max_steps):
        """
        长期奖励：基于最终检索质量和步数效率
        
        参数:
            final_quality (Tensor): 最终检索结果质量 (batch,)
            step_count (Tensor): 当前步数 (batch,)
            max_steps (float): 最大步数
            
        返回:
            Tensor: 长期奖励 (batch,)
        """
        reward = self.quality_weight * final_quality
        reward += self.early_stop_bonus * (1 - step_count / max_steps)
        return reward

    def correlation(self, x, y):
        """
        计算两个张量之间的相关系数
        
        参数:
            x, y (Tensor): 形状为 (batch,)
            
        返回:
            标量：相关系数
        """
        x_mean = x.mean()
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).mean()
        std_x = x.std() + 1e-8
        std_y = y.std() + 1e-8
        return cov / (std_x * std_y)
    
    def compute_orthogonality_loss(self, r_short, r_mid, r_long):
        """
        计算奖励分量之间的正交性损失，目标是降低奖励信号之间的相关性
        
        参数:
            r_short, r_mid, r_long (Tensor): 各奖励分量 (batch,)
            
        返回:
            Tensor: 正交性损失（标量）
        """
        corr_sm = self.correlation(r_short, r_mid)
        corr_sl = self.correlation(r_short, r_long)
        corr_ml = self.correlation(r_mid, r_long)
        loss = self.ortho_weight * (corr_sm**2 + corr_sl**2 + corr_ml**2)
        return loss

    def forward(self, current_quality, previous_quality, uncertainty, action, step_count, max_steps, final_quality=None):
        """
        计算分层奖励。如果提供 final_quality，则计算长期奖励，否则长期奖励置零。
        
        参数:
            current_quality (Tensor): 当前质量 (batch,)
            previous_quality (Tensor): 上一步质量 (batch,)
            uncertainty (Tensor): 当前不确定性 (batch,)
            action (Tensor): 动作 (batch,) 取值0或1
            step_count (Tensor): 当前步数 (batch,)
            max_steps (float): 最大步数
            final_quality (Tensor, optional): 最终质量 (batch,)
            
        返回:
            total_reward (Tensor): 综合奖励 (batch,)
            ortho_loss (Tensor): 正交性损失（标量）
            individual_rewards (dict): 各层奖励字典，包括短期、中期、长期奖励
        """
        # 计算短期奖励
        r_short = self.compute_short_term_reward(current_quality, previous_quality)
        
        # 计算中期奖励：对batch内每个样本逐个计算
        r_mid_list = []
        for u, a, s in zip(uncertainty, action, step_count):
            r_mid_list.append(self.compute_mid_term_reward(u.item(), a.item(), s.item(), max_steps))
        r_mid = torch.tensor(r_mid_list, device=current_quality.device)
        
        # 计算长期奖励
        if final_quality is not None:
            r_long = self.compute_long_term_reward(final_quality, step_count, max_steps)
        else:
            r_long = torch.zeros_like(r_short)
        
        # 综合奖励：动态权重加权求和
        total_reward = self.alpha * r_short + self.beta * r_mid + self.gamma * r_long
        
        # 计算正交性损失，鼓励各层奖励之间低相关性
        ortho_loss = self.compute_orthogonality_loss(r_short, r_mid, r_long)
        
        individual_rewards = {'short_term': r_short, 'mid_term': r_mid, 'long_term': r_long}
        
        return total_reward, ortho_loss, individual_rewards

def quality_metric(predictions, ground_truth, metric_type='accuracy'):
    """
    计算检索结果的质量指标。
    
    参数:
        predictions: 模型预测结果
        ground_truth: 真实标签
        metric_type (str): 指标类型 ('accuracy', 'recall@k', 'vqa', etc.)
        
    返回:
        float: 质量分数（范围在0-1之间）
    """
    if metric_type == 'accuracy':
        # 分类任务
        if isinstance(predictions, torch.Tensor) and isinstance(ground_truth, torch.Tensor):
            return (predictions == ground_truth).float().mean().item()
        else:
            return np.mean(np.array(predictions) == np.array(ground_truth))
    
    elif metric_type == 'recall@k':
        # 检索任务，默认 top-10
        k = min(len(predictions), 10)
        hits = 0
        for gt in ground_truth:
            if gt in predictions[:k]:
                hits += 1
        return hits / len(ground_truth)
    
    elif metric_type == 'vqa':
        # VQA任务：至少3个标注者给出相同答案得1分
        score = 0
        for pred, gts in zip(predictions, ground_truth):
            gt_counts = {}
            for gt in gts:
                gt_counts[gt] = gt_counts.get(gt, 0) + 1
            max_count = max(gt_counts.values()) if gt_counts else 0
            score += min(max_count / 3, 1)
        return score / len(predictions)
    
    else:
        return 0.5  # 默认返回0.5

