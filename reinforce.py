# Implementing the REINFORCE WITH BASELINE algorithm

import torch 
import numpy as np
from torch.distributions import Categorical
from models import PolicyNetwork, ValueNetwork
from env_cache import CacheEnv
from env_cat_monsters import CatMonstersEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def sample_episode(env, policy_net: PolicyNetwork, device: torch.device, gamma: float = 0.99) -> tuple:
    """
    Sample an episode using the current policy.
    
    Args:
        env: Environment (any gym-like environment)
        policy_net: Policy network
        device: torch device
        gamma: discount factor (unused, kept for compatibility)
    
    Returns:
        states, actions, rewards, and episode statistics
    """
    states, actions, rewards = [], [], []
    
    state = env.reset()
    done = False

    while not done:
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_logits = policy_net(s_tensor)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        
        next_state, reward, done, _ = env.step(action.item())
        
        states.append(s_tensor)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Compute episode statistics
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards) if rewards else 0.0
    
    episode_stats = {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'num_steps': len(rewards)
    }
    
    return states, actions, rewards, episode_stats

def compute_returns(rewards: list, gamma: float = 0.99) -> list:
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def reinforce_update(policy_net: PolicyNetwork, value_net: ValueNetwork, optimiser_policy: torch.optim.Optimizer, 
                     optimiser_value: torch.optim.Optimizer, states: list, actions: list, rewards: list, gamma: float = 0.99, 
                     device: torch.device = torch.device('cpu'), normalize_advantages: bool = True, 
                     entropy_coef: float = 0.01, max_grad_norm: float = 0.5) -> tuple:
  
    returns = torch.tensor(compute_returns(rewards, gamma),
                           dtype=torch.float32, device=device)
    states_tensor = torch.stack(states).to(device)
    actions_tensor = torch.tensor([a.item() for a in actions],
                                  dtype=torch.long, device=device)

    # Value network update
    values = value_net(states_tensor)
    # Ensure proper shape handling
    if len(values.shape) == 0:
        values = values.unsqueeze(0)
    elif len(values.shape) > 1:
        values = values.squeeze()
    critic_loss = F.mse_loss(values, returns)

    # Policy network update
    logits = policy_net(states_tensor)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    chosen_log_probs = log_probs[torch.arange(len(actions_tensor), device=device), actions_tensor]

    # Compute advantages
    advantages = (returns - values.detach())
    
    # Normalize advantages to reduce variance (critical for stable learning)
    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Policy loss with entropy bonus for exploration
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    actor_loss = -(chosen_log_probs * advantages).mean() - entropy_coef * entropy

    # Update policy network with gradient clipping
    optimiser_policy.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    optimiser_policy.step()

    # Update value network with gradient clipping
    optimiser_value.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
    optimiser_value.step()

    return actor_loss.item(), critic_loss.item(), entropy.item()

def train_reinforce(env, policy_net: PolicyNetwork, value_net: ValueNetwork, optimiser_policy: torch.optim.Optimizer, 
                    optimiser_value: torch.optim.Optimizer, device: torch.device = torch.device('cpu'), num_episodes: int = 1000, 
                    gamma: float = 0.99, normalize_advantages: bool = True,
                    entropy_coef: float = 0.01, max_grad_norm: float = 0.5):
    """
    Train REINFORCE algorithm on any environment.
    
    Args:
        env: Environment (any gym-like environment)
        policy_net: Policy network
        value_net: Value network
        optimiser_policy: Optimizer for policy network
        optimiser_value: Optimizer for value network
        device: torch device
        num_episodes: Number of training episodes
        gamma: Discount factor
        normalize_advantages: Whether to normalize advantages
        entropy_coef: Coefficient for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        episode_rewards: List of total rewards per episode
        episode_losses: List of (actor_loss, critic_loss, entropy) tuples per episode
    """
    episode_rewards = []
    episode_losses = []
    
    for _ in range(num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, gamma)
        actor_loss, critic_loss, entropy = reinforce_update(policy_net, value_net, optimiser_policy, optimiser_value, 
                                                               states, actions, rewards, gamma, device, normalize_advantages,
                                                               entropy_coef, max_grad_norm)
        
        episode_rewards.append(stats['total_reward'])
        episode_losses.append((actor_loss, critic_loss, entropy))
        
    return episode_rewards, episode_losses
    

def eval_cache_env(env: CacheEnv, policy_net: PolicyNetwork, device: torch.device, num_episodes: int = 100) -> dict:
    """
    Evaluate policy on CacheEnv.
    
    Args:
        env: CacheEnv instance
        policy_net: Policy network
        device: torch device
        num_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_steps = []
    hit_rates = []
    
    for _ in range(num_episodes):
        _, _, rewards, stats = sample_episode(env, policy_net, device)
        episode_rewards.append(stats['total_reward'])
        episode_steps.append(stats['num_steps'])
        
        # Calculate hit rate for cache environment
        num_hits = sum(1 for r in rewards if r == 1)
        hit_rate = num_hits / len(rewards) if len(rewards) > 0 else 0.0
        hit_rates.append(hit_rate)
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'avg_hit_rate': np.mean(hit_rates),
        'std_hit_rate': np.std(hit_rates),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'hit_rates': hit_rates
    }

def eval_cat_monsters_env(env: CatMonstersEnv, policy_net: PolicyNetwork, device: torch.device, num_episodes: int = 100) -> dict:
    """
    Evaluate policy on CatMonstersEnv.
    
    Args:
        env: CatMonstersEnv instance
        policy_net: Policy network
        device: torch device
        num_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_steps = []
    monster_hits = []
    reached_food_count = 0
    
    for _ in range(num_episodes):
        _, _, rewards, stats = sample_episode(env, policy_net, device)
        episode_rewards.append(stats['total_reward'])
        episode_steps.append(stats['num_steps'])
        
        # Count monster hits
        num_monster_hits = sum(1 for r in rewards if r == -8.0)
        monster_hits.append(num_monster_hits)
        
        # Check if reached food (episode always terminates at food)
        reached_food_count += 1
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'avg_monster_hits': np.mean(monster_hits),
        'std_monster_hits': np.std(monster_hits),
        'reached_food_rate': reached_food_count / num_episodes,
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'monster_hits': monster_hits
    }


def reinforce_train_cat_monsters():
    """Train REINFORCE algorithm on Cat and Monsters environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment parameters
    env = CatMonstersEnv(seed=42)
    state_dim = 25  # 5x5 grid, one-hot encoded
    action_dim = 4  # 4 actions: AU, AD, AL, AR
    
    policy_net = PolicyNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)
    value_net = ValueNetwork(state_dim=state_dim, hidden_dim=128).to(device)
    
    optimiser_policy = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
    optimiser_value = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    # Setup plot directory
    plot_dir = "plots/cat_monsters_reinforce"
    os.makedirs(plot_dir, exist_ok=True)
    
    episode_rewards = []
    episode_steps = []
    actor_losses = []
    critic_losses = []
    entropies = []
    
    num_episodes = 3000
    
    for _ in range(num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device)
        actor_loss, critic_loss, entropy = reinforce_update(policy_net, value_net, optimiser_policy, optimiser_value, 
                                                               states, actions, rewards, gamma=0.925, device=device, 
                                                               normalize_advantages=True, entropy_coef=0.01, max_grad_norm=0.5)
        print(f"Episode {_ + 1}/{num_episodes} - Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
        episode_rewards.append(stats['total_reward'])
        episode_steps.append(stats['num_steps'])
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        entropies.append(entropy)
    
    # Save the policy and value networks
    torch.save(policy_net.state_dict(), "policy_net_cat_monsters.pth")
    torch.save(value_net.state_dict(), "value_net_cat_monsters.pth")
    
    # Plot learning curves
    episodes = np.arange(1, num_episodes + 1)
    window = 100
    
    # Plot 1: Episode Reward
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg_reward = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_reward, color='red', linewidth=2, label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'episode_reward.png'), dpi=150)
    plt.close()
    
    # Plot 2: Episode Steps
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_steps, alpha=0.3, color='green', label='Episode Steps')
    if len(episode_steps) >= window:
        moving_avg_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_steps, color='orange', linewidth=2, label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Episode Steps Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'episode_steps.png'), dpi=150)
    plt.close()
    
    # Plot 3: Losses
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, actor_losses, alpha=0.3, color='purple', label='Actor Loss')
    plt.plot(episodes, critic_losses, alpha=0.3, color='brown', label='Critic Loss')
    if len(actor_losses) >= window:
        moving_avg_actor = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
        moving_avg_critic = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_actor, color='purple', linewidth=2, label=f'Actor Loss MA ({window})')
        plt.plot(episodes[window-1:], moving_avg_critic, color='brown', linewidth=2, label=f'Critic Loss MA ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_losses.png'), dpi=150)
    plt.close()
    
    # Plot 4: Entropy
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, entropies, alpha=0.3, color='teal', label='Policy Entropy')
    if len(entropies) >= window:
        moving_avg_entropy = np.convolve(entropies, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_entropy, color='darkblue', linewidth=2, label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'policy_entropy.png'), dpi=150)
    plt.close()
    
    # Plot 5: Combined metrics
    _, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward
    axes[0, 0].plot(episodes, episode_rewards, alpha=0.2, color='blue')
    if len(episode_rewards) >= window:
        moving_avg_reward = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg_reward, color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Steps
    axes[0, 1].plot(episodes, episode_steps, alpha=0.2, color='green')
    if len(episode_steps) >= window:
        moving_avg_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], moving_avg_steps, color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Number of Steps')
    axes[0, 1].set_title('Episode Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    axes[1, 0].plot(episodes, actor_losses, alpha=0.2, color='purple', label='Actor')
    axes[1, 0].plot(episodes, critic_losses, alpha=0.2, color='brown', label='Critic')
    if len(actor_losses) >= window:
        moving_avg_actor = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
        moving_avg_critic = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(episodes[window-1:], moving_avg_actor, color='purple', linewidth=2)
        axes[1, 0].plot(episodes[window-1:], moving_avg_critic, color='brown', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 1].plot(episodes, entropies, alpha=0.2, color='teal')
    if len(entropies) >= window:
        moving_avg_entropy = np.convolve(entropies, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(episodes[window-1:], moving_avg_entropy, color='darkblue', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'all_metrics.png'), dpi=150)
    plt.close()

def load_policy_and_value_networks():
    policy_net = PolicyNetwork(state_dim=25, action_dim=4, hidden_dim=128)
    value_net = ValueNetwork(state_dim=25, hidden_dim=128)
    policy_net.load_state_dict(torch.load("policy_net_cat_monsters.pth"))
    value_net.load_state_dict(torch.load("value_net_cat_monsters.pth"))
    return policy_net, value_net

def test_policy_and_value_networks():
    """Test loaded policy and value networks on CatMonstersEnv."""
    policy_net, value_net = load_policy_and_value_networks()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    value_net.to(device)
    env = CatMonstersEnv(seed=42)
    
    eval_results = eval_cat_monsters_env(env, policy_net, device, num_episodes=100)
    return eval_results

if __name__ == "__main__":
    # test_sample_episode()
    # smoke_test()  # Uncomment to test cache environment
    reinforce_train_cat_monsters()  # Train REINFORCE algorithm on Cat and Monsters environment
    test_policy_and_value_networks()


