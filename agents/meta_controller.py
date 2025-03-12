import torch
import torch.nn as nn

class MetaController(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MetaController, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, query_emb, result_emb, uncertainty):
        state = torch.cat([query_emb, result_emb, uncertainty.unsqueeze(-1)], dim=-1)  # 确保维度匹配
        encoded_state = self.state_encoder(state)
        action_probs = self.policy(encoded_state)
        value = self.value(encoded_state)
        return action_probs, value

    def act(self, query_emb, result_emb, uncertainty, quality, uncertainty_threshold=0.3, quality_threshold=0.8, deterministic=False):
        action_probs, _ = self.forward(query_emb, result_emb, uncertainty)
       
        if uncertainty.item() < uncertainty_threshold and quality > quality_threshold:
            return 0  # 停止
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        return action.item()