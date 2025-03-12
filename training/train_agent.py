import os
import torch
import numpy as np
from tqdm import tqdm
import logging
from recursive_retrieval.agents.environment import RecursiveRetrievalEnvironment
from recursive_retrieval.agents.rl.ppo import PPOTrainer
def train_agent(env, ppo_trainer, num_episodes=1000, max_steps=10, eval_freq=50, save_path='checkpoints'):
    """Train the meta-controller agent using PPO."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Training metrics
    best_eval_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    
    for episode in tqdm(range(num_episodes)):
        # Reset environment
        state = env.reset()
        
        # Storage for PPO update
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Store current state
            states.append(state)
            
            # Sample action from policy
            query_emb = state['query_emb'].unsqueeze(0)
            result_emb = state['result_emb'].unsqueeze(0)
            uncertainty = torch.tensor([state['uncertainty']]).unsqueeze(0).unsqueeze(0)
            
            action_probs, _ = ppo_trainer.meta_controller(query_emb, result_emb, uncertainty)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store step data
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Update PPO
        ppo_trainer.update(states, actions, log_probs, rewards, dones)
        
        # Log metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Periodically log and evaluate
        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(episode_rewards[-eval_freq:])
            avg_length = np.mean(episode_lengths[-eval_freq:])
            
            logger.info(f"Episode {episode+1}, Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.2f}")
            
            # Evaluate the agent
            eval_reward = evaluate_agent(env, ppo_trainer.meta_controller, num_episodes=20)
            logger.info(f"Evaluation Reward: {eval_reward:.3f}")
            
            # Save if best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(ppo_trainer.meta_controller.state_dict(), os.path.join(save_path, 'best_agent.pth'))
                logger.info("Saved new best model")
                
        # Save checkpoint periodically
        if (episode + 1) % 100 == 0:
            torch.save({
                'episode': episode,
                'meta_controller': ppo_trainer.meta_controller.state_dict(),
                'optimizer': ppo_trainer.optimizer.state_dict(),
                'best_eval_reward': best_eval_reward
            }, os.path.join(save_path, f'checkpoint_{episode+1}.pth'))
            
    return episode_rewards, episode_lengths

def evaluate_agent(env, meta_controller, num_episodes=20):
    """Evaluate the agent's performance."""
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action deterministically
            query_emb = state['query_emb'].unsqueeze(0)
            result_emb = state['result_emb'].unsqueeze(0)
            uncertainty = torch.tensor([state['uncertainty']]).unsqueeze(0).unsqueeze(0)
            
            action = meta_controller.act(query_emb, result_emb, uncertainty, deterministic=True)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update reward and state
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)

if __name__ == "__main__":
   
    from recursive_retrieval.models.recursive_model import RecursiveRetriever
    from recursive_retrieval.agents.meta_controller import MetaController
 
    config = {...}  
    recursive_model = RecursiveRetriever(config)
    meta_controller = MetaController(embedding_dim=512, hidden_dim=256)
    ppo_trainer = PPOTrainer(meta_controller)
    env = RecursiveRetrievalEnvironment(recursive_model, dataset, meta_controller)
    rewards, lengths = train_agent(env, ppo_trainer)