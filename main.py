import argparse
import yaml
import torch
import os
from models.base_retrieval.blip_model import BLIPRetriever
from models.uncertainty.estimator import UncertaintyEstimator
from models.refinement.recursive_model import RecursiveRefinementModel
from agents.meta_controller import MetaController
from agents.environment import RecursiveRetrievalEnvironment
from rl.ppo import PPOTrainer
from training.train_agent import train_agent
from data.datasets.infoseek import InfoseekDataset
from data.datasets.flickr30k import Flickr30kDataset
from data.datasets.vqa import VQADataset

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset_name = config['dataset']['name']
    if dataset_name == 'infoseek':
        dataset = InfoseekDataset(config['dataset']['path'])
    elif dataset_name == 'flickr30k':
        dataset = Flickr30kDataset(config['dataset']['path'])
    elif dataset_name == 'vqa':
        dataset = VQADataset(config['dataset']['path'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create base retriever
    base_retriever = BLIPRetriever(config['model'])
    
    # Create uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['uncertainty_hidden_dim']
    ).to(device)
    
    # Create recursive refinement model
    recursive_model = RecursiveRefinementModel(
        base_retriever=base_retriever,
        uncertainty_estimator=uncertainty_estimator,
        embedding_dim=config['model']['embedding_dim']
    ).to(device)
    
    # Create meta-controller
    meta_controller = MetaController(
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['agent']['hidden_dim']
    ).to(device)
    
    # Create environment
    env = RecursiveRetrievalEnvironment(
        recursive_model=recursive_model,
        dataset=dataset,
        target_metric=config['training']['target_metric']
    )
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        meta_controller=meta_controller,
        lr=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        epsilon=config['training']['epsilon'],
        value_coef=config['training']['value_coef'],
        entropy_coef=config['training']['entropy_coef']
    )
    
    # Train agent
    train_agent(
        env=env,
        ppo_trainer=ppo_trainer,
        num_episodes=config['training']['num_episodes'],
        max_steps=config['training']['max_steps'],
        eval_freq=config['training']['eval_frequency'],
        save_path=config['training']['save_path']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Recursive Self-Refinement Retrieval")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)