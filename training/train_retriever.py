import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from models.recursive_model import RecursiveRetriever
from models.uncertainty import UncertaintyEstimator
from data.dataloader import get_dataloader
from utils.metrics import compute_metrics

class RetrieverTrainer:
    def __init__(self, config):
        """
        初始化检索器训练器
        
        参数:
            config (dict): 配置文件
        """
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.recursive_model = RecursiveRetriever(config).to(self.device)
        
        # 优化器配置
        self.optimizer = optim.Adam(
            self.recursive_model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # 不确定性估计器优化器
        self.uncertainty_optimizer = optim.Adam(
            self.recursive_model.uncertainty_estimator.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # 奖励函数优化器
        self.reward_optimizer = optim.Adam(
            self.recursive_model.reward_function.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.recursive_model.train()
        total_loss = 0
        total_metrics = {}
        total_uncertainty_loss = 0
        total_reward_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # 1. 训练不确定性估计器
            self.uncertainty_optimizer.zero_grad()
            uncertainty_loss = self._train_uncertainty_estimator(batch)
            uncertainty_loss.backward()
            self.uncertainty_optimizer.step()
            total_uncertainty_loss += uncertainty_loss.item()
            
            # 2. 训练奖励函数
            self.reward_optimizer.zero_grad()
            reward_loss = self._train_reward_function(batch)
            reward_loss.backward()
            self.reward_optimizer.step()
            total_reward_loss += reward_loss.item()
            
            # 3. 训练主模型
            self.optimizer.zero_grad()
            results, history = self.recursive_model.retrieve(batch['query'], batch['candidates'])
            loss = self._compute_loss(results, batch['targets'])
            loss.backward()
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            metrics = compute_metrics(results, batch['targets'])
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
        
        # 计算平均损失和指标
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_uncertainty_loss = total_uncertainty_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {
            'loss': avg_loss,
            'uncertainty_loss': avg_uncertainty_loss,
            'reward_loss': avg_reward_loss,
            'metrics': avg_metrics
        }
    
    def _train_uncertainty_estimator(self, batch):
        """训练不确定性估计器"""
        query_emb = self.recursive_model.base_retriever.encode_text(batch['query'])
        target_emb = self.recursive_model.base_retriever.encode_text(batch['targets'])
        
        # 计算对抗扰动
        uncertainty = self.recursive_model.uncertainty_estimator(
            query_emb,
            target_emb
        )
        
        # 计算互信息流损失
        flow_loss = self.recursive_model.uncertainty_estimator.mutual_information_flow(
            query_emb,
            target_emb
        )
        
        return uncertainty.mean() + flow_loss
    
    def _train_reward_function(self, batch):
        """训练奖励函数"""
        results, history = self.recursive_model.retrieve(batch['query'], batch['candidates'])
        
        # 计算基础奖励
        rewards = self.recursive_model.reward_function(results)
        
        # 计算正交性损失
        orthogonality_loss = self.recursive_model.reward_function.compute_orthogonality_loss()
        
        return -rewards.mean() + orthogonality_loss
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.recursive_model.eval()
        total_loss = 0
        total_metrics = {}
        total_uncertainty_loss = 0
        total_reward_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # 获取批次数据
                query = batch['query']
                candidates = batch['candidates']
                targets = batch['targets']
                
                # 前向传播
                results, history = self.recursive_model.retrieve(query, candidates)
                
                # 计算损失和指标
                loss = self._compute_loss(results, targets)
                uncertainty_loss = self._train_uncertainty_estimator(batch)
                reward_loss = self._train_reward_function(batch)
                metrics = compute_metrics(results, targets)
                
                # 更新统计信息
                total_loss += loss.item()
                total_uncertainty_loss += uncertainty_loss.item()
                total_reward_loss += reward_loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
        
        # 计算平均值
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_uncertainty_loss = total_uncertainty_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {
            'loss': avg_loss,
            'uncertainty_loss': avg_uncertainty_loss,
            'reward_loss': avg_reward_loss,
            'metrics': avg_metrics
        }
    
    def _compute_loss(self, results, targets):
        """计算损失函数"""
        # 计算检索损失（基于排序的损失）
        retrieval_loss = self._compute_retrieval_loss(results, targets)
        
        # 计算不确定性损失（鼓励模型在不确定时停止）
        uncertainty_loss = self._compute_uncertainty_loss(results)
        
        # 总损失
        total_loss = retrieval_loss + self.config['training']['uncertainty_weight'] * uncertainty_loss
        
        return total_loss
    
    def _compute_retrieval_loss(self, results, targets):
        """计算检索损失"""
        # 使用 InfoNCE 损失
        query_emb = self.recursive_model.base_retriever.encode_text(results['query'])
        target_emb = self.recursive_model.base_retriever.encode_text(targets)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(query_emb, target_emb.t())
        
        # InfoNCE 损失
        loss = nn.CrossEntropyLoss()(sim_matrix, torch.arange(len(targets)).to(self.device))
        
        return loss
    
    def _compute_uncertainty_loss(self, results):
        """计算不确定性损失"""
        # 使用不确定性估计器的输出作为损失
        uncertainty = self.recursive_model.uncertainty_estimator(
            results['query_emb'],
            results['result_emb']
        )
        
        return uncertainty.mean()
    
    def save_checkpoint(self, path, epoch, metrics):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.recursive_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'uncertainty_optimizer_state_dict': self.uncertainty_optimizer.state_dict(),
            'reward_optimizer_state_dict': self.reward_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.recursive_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.uncertainty_optimizer.load_state_dict(checkpoint['uncertainty_optimizer_state_dict'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

def train_retriever(config):
    """训练递归检索模型的主函数"""
    # 创建训练器
    trainer = RetrieverTrainer(config)
    
    # 创建数据加载器
    train_loader = get_dataloader(
        config['dataset']['name'],
        'train',
        config,
        batch_size=config['training']['batch_size']
    )
    val_loader = get_dataloader(
        config['dataset']['name'],
        'val',
        config,
        batch_size=config['training']['batch_size']
    )
    
    # 训练循环
    best_metric = 0
    for epoch in range(config['training']['num_epochs']):
        # 训练一个epoch
        train_results = trainer.train_epoch(train_loader)
        
        # 评估
        val_results = trainer.evaluate(val_loader)
        
        # 更新学习率
        trainer.scheduler.step(val_results['metrics'][config['training']['target_metric']])
        
        # 记录日志
        trainer.logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        trainer.logger.info(f"Train Loss: {train_results['loss']:.4f}, Val Loss: {val_results['loss']:.4f}")
        trainer.logger.info(f"Train Uncertainty Loss: {train_results['uncertainty_loss']:.4f}")
        trainer.logger.info(f"Train Reward Loss: {train_results['reward_loss']:.4f}")
        trainer.logger.info(f"Train Metrics: {train_results['metrics']}")
        trainer.logger.info(f"Val Metrics: {val_results['metrics']}")
        
        # 保存最佳模型
        current_metric = val_results['metrics'][config['training']['target_metric']]
        if current_metric > best_metric:
            best_metric = current_metric
            trainer.save_checkpoint(
                os.path.join(config['training']['save_path'], 'best_model.pth'),
                epoch,
                val_results
            )
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            trainer.save_checkpoint(
                os.path.join(config['training']['save_path'], f'checkpoint_epoch_{epoch+1}.pth'),
                epoch,
                val_results
            )

if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open("config/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 开始训练
    train_retriever(config)
