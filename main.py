import argparse
import yaml
import torch
import os
from training.train_retriever import train_retriever
from training.train_agent import train_agent
from models.base_retrieval.blip_model import BLIPRetriever
from models.uncertainty.estimator import UncertaintyEstimator
from models.refinement.recursive_model import RecursiveRefinementModel
from agents.meta_controller import MetaController
from agents.environment import RecursiveRetrievalEnvironment
from rl.ppo import PPOTrainer
from data.datasets.infoseek import InfoseekDataset
from data.datasets.flickr30k import Flickr30kDataset
from data.datasets.vqa import VQADataset
from data.viquae_dataset.dataset import ViQuAEDataset


def train_retriever_mode(config):
    """训练检索模型模式"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    # 创建保存目录
    os.makedirs(config['training']['save_path'], exist_ok=True)
    
    # 开始训练
    train_retriever(config)

def train_agent_mode(config):
    """训练agent模式"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    dataset_name = config['dataset']['name']
    if dataset_name == 'infoseek':
        dataset = InfoseekDataset(config['dataset']['path'])
    elif dataset_name == 'flickr30k':
        dataset = Flickr30kDataset(config['dataset']['path'])
    elif dataset_name == 'vqa':
        dataset = VQADataset(config['dataset']['path'])
    elif dataset_name == 'viquae':
        dataset = ViQuAEDataset(
            jsonl_file=config['dataset']['jsonl_file'],
            image_dir=config['dataset']['image_dir'],
            split=config['dataset'].get('split', 'train')
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 创建基础检索器
    base_retriever = BLIPRetriever(config['model'])
    
    # 创建不确定性估计器
    uncertainty_estimator = UncertaintyEstimator(
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['uncertainty_hidden_dim']
    ).to(device)
    
    # 创建递归优化模型
    recursive_model = RecursiveRefinementModel(
        base_retriever=base_retriever,
        uncertainty_estimator=uncertainty_estimator,
        embedding_dim=config['model']['embedding_dim']
    ).to(device)
    
    # 创建元控制器
    meta_controller = MetaController(
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['agent']['hidden_dim']
        belief_dim=config['agent'].get('belief_dim', 64)  
    ).to(device)
    
    # 创建环境
    env = RecursiveRetrievalEnvironment(
        recursive_model=recursive_model,
        dataset=dataset,
        target_metric=config['training']['target_metric']
    )
    
    # 创建PPO训练器
    ppo_trainer = PPOTrainer(
        meta_controller=meta_controller,
        lr=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        epsilon=config['training']['epsilon'],
        value_coef=config['training']['value_coef'],
        entropy_coef=config['training']['entropy_coef']
    )
    
    # 训练agent
    train_agent(
        env=env,
        ppo_trainer=ppo_trainer,
        num_episodes=config['training']['num_episodes'],
        max_steps=config['training']['max_steps'],
        eval_freq=config['training']['eval_frequency'],
        save_path=config['training']['save_path']
    )

def main(config_path, mode='retriever'):
    """
    主函数：加载配置并开始训练
    
    参数:
        config_path (str): 配置文件路径
        mode (str): 训练模式，'retriever' 或 'agent'
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 根据模式选择训练函数
    if mode == 'retriever':
        train_retriever_mode(config)
    elif mode == 'agent':
        train_agent_mode(config)
    else:
        raise ValueError(f"Unknown training mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练递归自优化检索模型")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--mode", type=str, default="retriever",
                       choices=['retriever', 'agent'],
                       help="训练模式：retriever（训练检索模型）或 agent（训练强化学习代理）")
    args = parser.parse_args()
    
    main(args.config, args.mode)