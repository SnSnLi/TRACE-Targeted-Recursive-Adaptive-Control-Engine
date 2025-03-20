
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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
            # -------------------------------
            # 1. 训练不确定性估计器
            # -------------------------------
            self.uncertainty_optimizer.zero_grad()
            uncertainty_loss = self._train_uncertainty_estimator(batch)
            uncertainty_loss.backward()
            self.uncertainty_optimizer.step()
            total_uncertainty_loss += uncertainty_loss.item()
            
            # -------------------------------
            # 2. 训练奖励函数 (如果使用RL/元控制器)
            # -------------------------------
            self.reward_optimizer.zero_grad()
            reward_loss = self._train_reward_function(batch)
            reward_loss.backward()
            self.reward_optimizer.step()
            total_reward_loss += reward_loss.item()
            
            
            # 3. 训练主检索模型
            
            self.optimizer.zero_grad()

            # 如果是 ViQuAE 任务，batch 里可能包含 "query" + "image"
            if isinstance(batch.get('query'), list) and isinstance(batch.get('image'), list):
                # 批次情况下 batch['query'] 和 batch['image'] 均为列表（长度等于 batch_size）
                # 这里简单起见，逐条处理
                combined_results = []
                combined_history = []
                for q_text, q_img, cand in zip(batch['query'], batch['image'], batch['candidates']):
                    # 将 (文本, 图像) 作为组合查询
                    res, hist = self.recursive_model.retrieve((q_text, q_img), cand)
                    combined_results.append(res)
                    combined_history.append(hist)
                # 如果 batch_size=1，可以直接取第一个
                # 否则需要自己处理多个结果，这里演示只取第一个
                results, history = combined_results[0], combined_history[0]
            else:
                # 常规检索（文本或图像单模态查询）
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
        
        # 计算不确定性
        uncertainty = self.recursive_model.uncertainty_estimator(query_emb, target_emb)
        
        # 计算互信息流损失
        flow_loss = self.recursive_model.uncertainty_estimator.mutual_information_flow(
            query_emb,
            target_emb
        )
        
        return uncertainty.mean() + flow_loss
    
    def _train_reward_function(self, batch):
        """训练奖励函数 (可选: 如果在强化学习中使用元控制器)"""
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
                query = batch['query']
                candidates = batch['candidates']
                targets = batch['targets']
                
                # ViQuAE 可能有图像
                if isinstance(query, list) and 'image' in batch:
                    # 只演示取 batch[0] 进行评估
                    results, history = self.recursive_model.retrieve((query[0], batch['image'][0]), candidates[0])
                else:
                    results, history = self.recursive_model.retrieve(query, candidates)
                
                # 计算损失和指标
                loss = self._compute_loss(results, targets)
                uncertainty_loss = self._train_uncertainty_estimator(batch)
                reward_loss = self._train_reward_function(batch)
                metrics = compute_metrics(results, targets)
                
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
        # 1. 检索损失（基于排序的 InfoNCE 损失）
        retrieval_loss = self._compute_retrieval_loss(results, targets)
        # 2. 不确定性损失（鼓励模型在不确定时停止检索）
        uncertainty_loss = self._compute_uncertainty_loss(results)
        
        # 加权总损失
        weight = self.config['training'].get('uncertainty_weight', 0.1)
        total_loss = retrieval_loss + weight * uncertainty_loss
        return total_loss
    
    def _compute_retrieval_loss(self, results, targets):
        """计算检索损失 (InfoNCE)"""
        # 根据 results 的结构获取查询文本
        if isinstance(results, dict) and 'query' in results:
            query_texts = results['query']
        else:
            # 可能是 results['query'] 也可能是直接列表
            # 这里仅演示一种简单情况
            query_texts = results['query'] if hasattr(results, 'get') else results
        
        query_emb = self.recursive_model.base_retriever.encode_text(query_texts)
        target_emb = self.recursive_model.base_retriever.encode_text(targets)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(query_emb, target_emb.t())
        
        # InfoNCE 损失
        device = self.device
        loss = nn.CrossEntropyLoss()(sim_matrix, torch.arange(len(targets)).to(device))
        return loss
    
    def _compute_uncertainty_loss(self, results):
        """计算不确定性损失"""
        if isinstance(results, dict) and 'query_emb' in results:
            q_emb = results['query_emb']
            r_emb = results['result_emb']
            uncertainty = self.recursive_model.uncertainty_estimator(q_emb, r_emb)
            return uncertainty.mean()
        else:
            # 如果没有预先计算的嵌入，则返回 0.0
            return torch.tensor(0.0, device=self.device)
    
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
    
    # 根据 dataset.name 判断是否为 viquae
    dataset_name = config['dataset']['name']
    batch_size = config['training']['batch_size']
    
    if dataset_name.lower() == 'viquae':
        # 如果数据集是 ViQuAE，则使用自定义数据加载逻辑
        train_file = config['dataset']['viquae']['jsonl_file']
        val_file = config['dataset']['viquae'].get('val_jsonl_file', None)
        image_dir = config['dataset']['viquae']['image_dir']
        
        class ViQuAEDataset(torch.utils.data.Dataset):
            def __init__(self, jsonl_path, image_dir):
                self.samples = []
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        question = item.get('input') or item.get('question') or item.get('query')
                        img_filename = item.get('image')
                        if question is None or img_filename is None:
                            continue
                        img_path = os.path.join(image_dir, img_filename)
                        try:
                            image = Image.open(img_path).convert('RGB')
                        except FileNotFoundError:
                            continue  # 如果图像缺失则跳过
                        
                        # 提取答案文本
                        answer = None
                        if 'output' in item:
                            # 有的版本output是列表，有的是dict
                            if isinstance(item['output'], dict):
                                answer = item['output'].get('original_answer') or item['output'].get('answer')
                            elif isinstance(item['output'], list) and len(item['output']) > 0:
                                answer = item['output'][0].get('answer') or item['output'][0].get('original_answer')
                        target_text = answer if answer else ""
                        
                        # 构建候选集（训练时使用批内负样本，这里仅包含自身target）
                        candidates = [target_text] if target_text else []
                        
                        self.samples.append({
                            'query': question,
                            'image': image,
                            'candidates': candidates,
                            'targets': target_text
                        })
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        train_dataset = ViQuAEDataset(train_file, image_dir)
        val_dataset = ViQuAEDataset(val_file, image_dir) if val_file else None
        
        def viquae_collate(batch):
            # 将同一批次的字段分别聚合到列表
            # batch: [sample, sample, ...]
            keys = batch[0].keys()
            collated = {k: [d[k] for d in batch] for k in keys}
            return collated
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=viquae_collate
        )
        
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=viquae_collate
            )
        else:
            val_loader = []
    else:
        # 如果不是viquae，则调用原始get_dataloader
        train_loader = get_dataloader(dataset_name, 'train', config, batch_size=batch_size)
        val_loader = get_dataloader(dataset_name, 'val', config, batch_size=batch_size)
    
    # 训练循环
    best_metric = 0
    for epoch in range(config['training']['num_epochs']):
        # 训练一个epoch
        train_results = trainer.train_epoch(train_loader)
        
        # 评估
        if val_loader:
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
        else:
            # 若无验证集，则仅输出训练信息
            trainer.logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            trainer.logger.info(f"Train Loss: {train_results['loss']:.4f}")
            trainer.logger.info(f"Train Uncertainty Loss: {train_results['uncertainty_loss']:.4f}")
            trainer.logger.info(f"Train Reward Loss: {train_results['reward_loss']:.4f}")
            trainer.logger.info(f"Train Metrics: {train_results['metrics']}")
    
if __name__ == "__main__":
    import yaml
    # 加载配置
    with open("config/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 开始训练
    train_retriever(config)

