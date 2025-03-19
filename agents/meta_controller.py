import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveBayesianMetaController(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, belief_dim):
        """
        初始化 Recursive Bayesian Stopping (RBS) 元控制器
        
        参数:
            embedding_dim (int): 输入嵌入维度
            hidden_dim (int): 隐藏层维度
            belief_dim (int): 信念分布维度（质量信念）
        """
        super(RecursiveBayesianMetaController, self).__init__()
        input_dim = embedding_dim * 2 + 1

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 信念分布参数（均值和对数方差）
        self.belief_mean = nn.Linear(hidden_dim // 2, belief_dim)
        self.belief_logvar = nn.Linear(hidden_dim // 2, belief_dim)

        # 停止概率和价值估计
        self.policy_head = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 初始信念（先验）
        self.prior_mean = torch.zeros(1, belief_dim)
        self.prior_logvar = torch.ones(1, belief_dim)

    def update_belief(self, mu, logvar, prev_mu, prev_logvar):
        """
        递归更新信念（贝叶斯更新）
        
        参数:
            mu, logvar: 当前观测分布参数
            prev_mu, prev_logvar: 前一步信念
        
        返回:
            更新后的 mu, logvar
        """
        precision = torch.exp(-logvar) + torch.exp(-prev_logvar)
        new_logvar = -torch.log(precision + 1e-8)
        new_mu = (mu * torch.exp(-logvar) + prev_mu * torch.exp(-prev_logvar)) / precision
        return new_mu, new_logvar

    def forward(self, query_emb, result_emb, uncertainty, prev_belief=None):
        """
        前向传播，递归更新信念
        
        参数:
            query_emb, result_emb, uncertainty: 输入状态
            prev_belief (tuple, optional): 前一步信念 (mu, logvar)
        
        返回:
            action_probs (Tensor): 停止概率 (batch, 2)
            value (Tensor): 价值估计 (batch, 1)
            belief (tuple): 更新后的信念 (mu, logvar)
        """
        state = torch.cat([query_emb, result_emb, uncertainty.unsqueeze(-1)], dim=-1)
        encoded_state = self.state_encoder(state)

        # 当前观测分布
        mu = self.belief_mean(encoded_state)
        logvar = self.belief_logvar(encoded_state)

        # 递归更新信念
        if prev_belief is None:
            prev_mu, prev_logvar = self.prior_mean.to(mu.device), self.prior_logvar.to(mu.device)
        else:
            prev_mu, prev_logvar = prev_belief
        belief_mu, belief_logvar = self.update_belief(mu, logvar, prev_mu, prev_logvar)

        # 基于信念生成动作和价值
        belief_sample = belief_mu + torch.randn_like(belief_mu) * torch.exp(0.5 * belief_logvar)
        action_probs = self.policy_head(belief_sample)
        value = self.value_head(belief_sample)

        return action_probs, value, (belief_mu, belief_logvar)

    def act(self, query_emb, result_emb, uncertainty, quality, prev_belief=None, uncertainty_threshold=0.3, quality_threshold=0.8, deterministic=False):
        """决策动作"""
        action_probs, _, belief = self.forward(query_emb, result_emb, uncertainty, prev_belief)
        if uncertainty.item() < uncertainty_threshold and quality > quality_threshold:
            return 0, belief
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        return action.item(), belief



