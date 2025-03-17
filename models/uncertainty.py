import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BlipProcessor, BlipForImageTextRetrieval

class UncertaintyEstimator(nn.Module):
    def __init__(self, embedding_dim, epsilon=1e-3):
        """
        Args:
            embedding_dim (int): 嵌入向量的维度。
            epsilon (float): 对抗扰动的步长大小。
        """
        super(UncertaintyEstimator, self).__init__()
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon  # 对抗扰动幅度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def compute_similarity(self, query_embedding, result_embedding):
        """
        计算查询和结果嵌入之间的余弦相似度。
        
        Args:
            query_embedding (torch.Tensor): 查询嵌入，形状 (batch, embedding_dim)
            result_embedding (torch.Tensor): 结果嵌入，形状 (batch, embedding_dim)
        
        Returns:
            torch.Tensor: 每个样本对的余弦相似度，形状 (batch,)
        """
        # 归一化
        query_norm = query_embedding / (query_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        result_norm = result_embedding / (result_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = (query_norm * result_norm).sum(dim=-1)
        return similarity

    def forward(self, query_embedding, result_embedding):
        """
        利用对抗扰动敏感性度量计算不确定性。
        思路：先计算原始余弦相似度，再通过对输入做小幅扰动（基于梯度方向），计算扰动后相似度，
        二者之差作为敏感性度量，高敏感性对应较高不确定性。
        
        Args:
            query_embedding (torch.Tensor): 查询嵌入，形状 (batch, embedding_dim)
            result_embedding (torch.Tensor): 结果嵌入，形状 (batch, embedding_dim)
        
        Returns:
            torch.Tensor: 每个样本的不确定性指标，归一化至 [0, 1]。
        """
        # 确保输入可求梯度
        query_embedding = query_embedding.clone().detach().to(self.device).requires_grad_(True)
        result_embedding = result_embedding.clone().detach().to(self.device).requires_grad_(True)
        
        # 计算原始余弦相似度
        baseline_similarity = self.compute_similarity(query_embedding, result_embedding)
        
        # 定义损失为负相似度（目标是使相似度下降，从而获得最大的变化）
        loss = -baseline_similarity.mean()
        loss.backward(retain_graph=True)
        
        # 获取梯度
        query_grad = query_embedding.grad
        result_grad = result_embedding.grad
        
        # 对梯度进行归一化后加上微小扰动
        query_adv = query_embedding + self.epsilon * query_grad / (query_grad.norm(dim=-1, keepdim=True) + 1e-8)
        result_adv = result_embedding + self.epsilon * result_grad / (result_grad.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 计算扰动后余弦相似度
        adv_similarity = self.compute_similarity(query_adv, result_adv)
        
        # 不确定性定义为扰动前后相似度的绝对变化值
        uncertainty = torch.abs(adv_similarity - baseline_similarity)
        
        # 归一化：由于余弦相似度范围在[-1, 1]，最大可能变化为2
        normalized_uncertainty = uncertainty / 2.0
        normalized_uncertainty = torch.clamp(normalized_uncertainty, 0, 1)
        
        return normalized_uncertainty.detach()
    
    def estimate(self, query, current_query, scores):
        """
        与其他检索模块的兼容接口，包装 forward 方法。
        """
        return self(query, current_query)



