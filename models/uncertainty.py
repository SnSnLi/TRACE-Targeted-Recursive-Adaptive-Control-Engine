import torch
import torch.nn as nn
import numpy as np
from transformers import BlipProcessor, BlipForImageTextRetrieval

class UncertaintyEstimator(nn.Module):
    def __init__(self, embedding_dim, num_samples=1000):
        super(UncertaintyEstimator, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_samples = num_samples  # 用于蒙特卡洛采样
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def estimate_joint_distribution(self, query_embedding, result_embedding):
        """
        Estimate joint probability distribution p(T, I) using Monte Carlo sampling.
        
        Args:
            query_embedding (torch.Tensor): Text embedding (T).
            result_embedding (torch.Tensor): Image embedding (I).
            
        Returns:
            torch.Tensor: Approximated joint probability distribution.
        """
        # Concatenate embeddings to form joint representation
        joint_features = torch.cat([query_embedding, result_embedding], dim=-1)  # [batch, 2 * embedding_dim]

        # Assume Gaussian distribution for simplicity (can be replaced with more complex models)
        mean = joint_features.mean(dim=0)
        covariance = torch.cov(joint_features.T) + 1e-6 * torch.eye(joint_features.size(-1)).to(self.device)

        # Sample from the joint distribution using Monte Carlo
        dist = torch.distributions.MultivariateNormal(mean, covariance)
        samples = dist.sample((self.num_samples,))  # [num_samples, 2 * embedding_dim]

        # Discretize samples into bins for probability estimation
        bins = 50
        hist, _ = torch.histogramdd(samples.cpu(), bins=bins, range=[[samples.min().item(), samples.max().item()]] * (2 * self.embedding_dim))
        joint_prob = hist / hist.sum()  # Normalize to get p(T, I)
        return joint_prob.to(self.device)

    def compute_joint_entropy(self, joint_prob):
        """
        Compute joint entropy H(T, I) = -∑ p(t, i) log p(t, i).
        
        Args:
            joint_prob (torch.Tensor): Joint probability distribution p(T, I).
            
        Returns:
            torch.Tensor: Joint entropy.
        """
        joint_prob = joint_prob.flatten()
        joint_prob = joint_prob[joint_prob > 0]  # Avoid log(0)
        joint_entropy = -torch.sum(joint_prob * torch.log(joint_prob))
        return joint_entropy

    def compute_marginal_entropy(self, features):
        """
        Compute marginal entropy for a single modality (H(T) or H(I)).
        
        Args:
            features (torch.Tensor): Embedding of a single modality.
            
        Returns:
            torch.Tensor: Marginal entropy.
        """
        # Discretize features into bins
        bins = 50
        hist, _ = torch.histogramdd(features.cpu(), bins=bins, range=[[features.min().item(), features.max().item()]] * self.embedding_dim)
        marginal_prob = hist / hist.sum()
        marginal_prob = marginal_prob.flatten()
        marginal_prob = marginal_prob[marginal_prob > 0]
        entropy = -torch.sum(marginal_prob * torch.log(marginal_prob))
        return entropy.to(self.device)

    def compute_conditional_entropy(self, joint_prob, marginal_prob):
        """
        Compute conditional entropy H(T|I) or H(I|T).
        
        Args:
            joint_prob (torch.Tensor): Joint probability distribution p(T, I).
            marginal_prob (torch.Tensor): Marginal probability p(I) or p(T).
            
        Returns:
            torch.Tensor: Conditional entropy.
        """
        # H(T|I) = H(T, I) - H(I)
        joint_entropy = self.compute_joint_entropy(joint_prob)
        marginal_entropy = self.compute_marginal_entropy(marginal_prob)
        return joint_entropy - marginal_entropy

    def compute_mutual_information(self, query_embedding, result_embedding, joint_prob):
        """
        Compute mutual information I(T; I) = H(T) + H(I) - H(T, I).
        
        Args:
            query_embedding (torch.Tensor): Text embedding (T).
            result_embedding (torch.Tensor): Image embedding (I).
            joint_prob (torch.Tensor): Joint probability distribution p(T, I).
            
        Returns:
            torch.Tensor: Mutual information.
        """
        H_T = self.compute_marginal_entropy(query_embedding)
        H_I = self.compute_marginal_entropy(result_embedding)
        H_TI = self.compute_joint_entropy(joint_prob)
        return H_T + H_I - H_TI

    def forward(self, query_embedding, result_embedding):
        """
        Estimate uncertainty using cross-modal joint entropy.
        
        Args:
            query_embedding (torch.Tensor): Query embedding (text or image).
            result_embedding (torch.Tensor): Result embedding (text or image).
            
        Returns:
            torch.Tensor: Joint entropy as uncertainty measure.
        """
        # Step 1: Estimate joint probability distribution p(T, I)
        joint_prob = self.estimate_joint_distribution(query_embedding, result_embedding)

        # Step 2: Compute joint entropy H(T, I)
        joint_entropy = self.compute_joint_entropy(joint_prob)

        # Step 3: Compute marginal entropies and mutual information (optional analysis)
        H_T = self.compute_marginal_entropy(query_embedding)
        H_I = self.compute_marginal_entropy(result_embedding)
        mutual_info = self.compute_mutual_information(query_embedding, result_embedding, joint_prob)
        H_T_given_I = self.compute_conditional_entropy(joint_prob, result_embedding)
        H_I_given_T = self.compute_conditional_entropy(joint_prob, query_embedding)

        # Step 4: Use joint entropy as uncertainty measure (normalize to [0, 1])
        max_entropy = (H_T + H_I).item()  # Maximum possible entropy if independent
        normalized_entropy = joint_entropy / max_entropy
        return torch.clamp(normalized_entropy, 0, 1)

    def estimate(self, query, current_query, scores):
        """
        Compatibility method for RecursiveRetriever.
        """
        return self(query, current_query)

