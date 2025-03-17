import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalMetaController(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, latent_dim):
        """
        初始化基于变分推断的元控制器
        
        参数:
            embedding_dim (int): 输入嵌入维度（query和result的维度）
            hidden_dim (int): 状态编码网络的隐藏层维度
            latent_dim (int): 潜变量 z 的维度
        """
        super(VariationalMetaController, self).__init__()
        # 输入为 query_emb, result_emb 和 uncertainty，共计 embedding_dim * 2 + 1
        input_dim = embedding_dim * 2 + 1
        
        # 状态编码器，将原始状态映射到隐藏表示
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # 通过隐藏表示生成潜变量分布的均值和对数方差
        self.latent_mean = nn.Linear(hidden_dim // 2, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # 基于潜变量 z 生成停止概率（两类动作：0-停止，1-继续）
        self.policy_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        # 基于潜变量 z 生成价值估计
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从 q(z|state)=N(mu, exp(logvar)) 中采样
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, query_emb, result_emb, uncertainty):
        """
        前向传播：
        1. 将输入状态（query_emb, result_emb, uncertainty）编码为隐藏表示
        2. 生成潜变量分布参数（均值和对数方差）
        3. 重参数化采样得到潜变量 z
        4. 由 z 生成动作概率和价值估计
        5. 计算 KL 散度作为 ELBO 的正则项（可作为训练时的辅助loss）
        
        返回:
            action_probs (Tensor): 停止/继续的概率分布，形状 (batch, 2)
            value (Tensor): 价值估计，形状 (batch, 1)
            elbo_loss (Tensor): 每个样本的 KL 损失均值
        """
        # 将 uncertainty 扩展为 (batch,1)，与 query_emb 和 result_emb 拼接
        state = torch.cat([query_emb, result_emb, uncertainty.unsqueeze(-1)], dim=-1)
        encoded_state = self.state_encoder(state)
        
        # 生成潜变量分布的均值和对数方差
        mu = self.latent_mean(encoded_state)
        logvar = self.latent_logvar(encoded_state)
        
        # 重参数化采样得到 z
        z = self.reparameterize(mu, logvar)
        
        # 通过 decoder 网络获得动作概率和价值估计
        action_probs = self.policy_decoder(z)
        value = self.value_head(z)
        
        # 计算 KL 散度，作为 q(z|state) 和标准正态分布 N(0,I) 之间的距离
        # KL_div = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) (对每个样本)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        elbo_loss = kl_div.mean()  # 平均到整个 batch
        
        return action_probs, value, elbo_loss

    def act(self, query_emb, result_emb, uncertainty, quality, uncertainty_threshold=0.3, quality_threshold=0.8, deterministic=False):
        """
        根据输入状态以及外部质量、不确定性阈值决定动作。
        如果不确定性较低且质量较高，则直接返回停止动作（0）。
        否则，根据动作概率决定是否停止或继续。
        
        参数:
            query_emb, result_emb, uncertainty: 输入状态
            quality (float): 当前的质量指标
            uncertainty_threshold (float): 不确定性阈值（低于此值倾向于停止）
            quality_threshold (float): 质量阈值（高于此值倾向于停止）
            deterministic (bool): 是否采用确定性决策（取概率最大值）
            
        返回:
            action (int): 0表示停止，1表示继续
        """
        action_probs, _, _ = self.forward(query_emb, result_emb, uncertainty)
        
        # 根据外部阈值进行提前决策
        if uncertainty.item() < uncertainty_threshold and quality > quality_threshold:
            return 0  # 直接停止
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        return action.item()



