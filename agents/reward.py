import torch
import numpy as np

class RewardFunction:
    """
    奖励函数类，用于计算递归检索过程中的奖励。
    根据检索结果的质量和不确定性来计算奖励，引导元控制器学习何时应该继续优化或停止。
    """
    
    def __init__(self, 
                 quality_weight=1.0, 
                 uncertainty_weight=0.5, 
                 step_penalty=0.1,
                 early_stop_bonus=0.5,
                 late_stop_penalty=0.3):
        """
        初始化奖励函数。
        
        参数:
            quality_weight (float): 结果质量的权重
            uncertainty_weight (float): 不确定性的权重
            step_penalty (float): 每步的惩罚（鼓励尽快完成）
            early_stop_bonus (float): 提前停止且结果好的奖励
            late_stop_penalty (float): 过晚停止的惩罚
        """
        self.quality_weight = quality_weight
        self.uncertainty_weight = uncertainty_weight
        self.step_penalty = step_penalty
        self.early_stop_bonus = early_stop_bonus
        self.late_stop_penalty = late_stop_penalty
    
    def compute_reward(self, 
                       current_quality, 
                       previous_quality, 
                       uncertainty, 
                       action, 
                       step_count, 
                       max_steps):
        """
        计算当前步骤的奖励。
        
        参数:
            current_quality (float): 当前检索结果的质量 (0-1)
            previous_quality (float): 上一步检索结果的质量 (0-1)
            uncertainty (float): 当前不确定性 (0-1)
            action (int): 采取的动作 (0: 停止, 1: 继续)
            step_count (int): 当前步数
            max_steps (int): 最大步数
            
        返回:
            float: 计算的奖励值
        """
        # 基础奖励：质量改进
        quality_improvement = current_quality - previous_quality
        quality_reward = self.quality_weight * quality_improvement
        
        # 步数惩罚
        step_penalty = -self.step_penalty * (step_count / max_steps)
        
        # 不确定性相关奖励
        uncertainty_reward = 0
        
        # 如果动作是停止 (0)
        if action == 0:
            # 如果不确定性低且质量高，奖励提前停止
            if uncertainty < 0.3 and current_quality > 0.7:
                uncertainty_reward += self.early_stop_bonus
            # 如果不确定性高且质量低，惩罚提前停止
            elif uncertainty > 0.7 and current_quality < 0.5:
                uncertainty_reward -= self.early_stop_bonus
        # 如果动作是继续 (1)
        else:
            # 如果不确定性高，奖励继续
            if uncertainty > 0.5:
                uncertainty_reward += self.uncertainty_weight * uncertainty
            # 如果不确定性低，惩罚继续
            else:
                uncertainty_reward -= self.uncertainty_weight * (1 - uncertainty)
            
            # 如果已经接近最大步数，增加惩罚
            if step_count > 0.8 * max_steps:
                uncertainty_reward -= self.late_stop_penalty
        
        # 总奖励
        total_reward = quality_reward + uncertainty_reward + step_penalty
        
        return total_reward
    
    def compute_final_reward(self, final_quality, step_count, max_steps):
        """
        计算整个检索过程的最终奖励。
        
        参数:
            final_quality (float): 最终检索结果的质量 (0-1)
            step_count (int): 总步数
            max_steps (int): 最大步数
            
        返回:
            float: 最终奖励值
        """
        # 基础奖励：最终质量
        quality_reward = self.quality_weight * final_quality
        
        # 步数效率奖励
        efficiency_bonus = self.early_stop_bonus * (1 - step_count / max_steps)
        
        # 总奖励
        total_reward = quality_reward + efficiency_bonus
        
        return total_reward


def quality_metric(predictions, ground_truth, metric_type='accuracy'):
    """
    计算检索结果的质量。
    
    参数:
        predictions: 模型预测结果
        ground_truth: 真实标签
        metric_type (str): 指标类型 ('accuracy', 'recall', 'f1', etc.)
        
    返回:
        float: 质量分数 (0-1)
    """
    if metric_type == 'accuracy':
        # 对于分类任务
        if isinstance(predictions, torch.Tensor) and isinstance(ground_truth, torch.Tensor):
            return (predictions == ground_truth).float().mean().item()
        else:
            return np.mean(np.array(predictions) == np.array(ground_truth))
    
    elif metric_type == 'recall@k':
        # 对于检索任务
        k = min(len(predictions), 10)  # 默认使用top-10
        hits = 0
        for gt in ground_truth:
            if gt in predictions[:k]:
                hits += 1
        return hits / len(ground_truth)
    
    elif metric_type == 'vqa':
        # 对于VQA任务的特殊评估
        score = 0
        for pred, gts in zip(predictions, ground_truth):
            # VQA评分：如果至少3个标注者给出相同答案，得分为1
            gt_counts = {}
            for gt in gts:
                gt_counts[gt] = gt_counts.get(gt, 0) + 1
            
            max_count = max(gt_counts.values()) if gt_counts else 0
            score += min(max_count / 3, 1)
        
        return score / len(predictions)
    
    else:
        # 默认返回0.5
        return 0.5
