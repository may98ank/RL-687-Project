import torch 
import numpy as np
from models import PolicyNetwork, ValueNetwork
from env_cat_monsters import CatMonstersEnv
import matplotlib.pyplot as plt
from reinforce import sample_episode, reinforce_update
import os



def reinforce_train_cat_monsters():
    """Train REINFORCE algorithm on Cat and Monsters environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment parameters
    env = CatMonstersEnv(seed=42)
    state_dim = 25  
    action_dim = 4  
    
    policy_net = PolicyNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)
    value_net = ValueNetwork(state_dim=state_dim, hidden_dim=128).to(device)
    
    optimiser_policy = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
    optimiser_value = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
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
    torch.save(policy_net.state_dict(), "checkpoints/cats_reinforce/policy_net_cat_monsters_reinforce.pth")
    torch.save(value_net.state_dict(), "checkpoints/cats_reinforce/value_net_cat_monsters_reinforce.pth")
    
    # Plot learning curves
    episodes = np.arange(1, num_episodes + 1)
    window = 100
    
    # Plot 1: Episode Reward
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg_reward = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(episodes[window - 1:], moving_avg_reward, color='red', linewidth=2, label=f'Moving Average ({window} episodes)')
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
        moving_avg_steps = np.convolve(episode_steps, np.ones(window) / window, mode='valid')
        plt.plot(episodes[window - 1:], moving_avg_steps, color='orange', linewidth=2, label=f'Moving Average ({window} episodes)')
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
        moving_avg_actor = np.convolve(actor_losses, np.ones(window) / window, mode='valid')
        moving_avg_critic = np.convolve(critic_losses, np.ones(window) / window, mode='valid')
        plt.plot(episodes[window - 1:], moving_avg_actor, color='purple', linewidth=2, label=f'Actor Loss MA ({window})')
        plt.plot(episodes[window - 1:], moving_avg_critic, color='brown', linewidth=2, label=f'Critic Loss MA ({window})')
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
        moving_avg_entropy = np.convolve(entropies, np.ones(window) / window, mode='valid')
        plt.plot(episodes[window - 1:], moving_avg_entropy, color='darkblue', linewidth=2, label=f'Moving Average ({window} episodes)')
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
        moving_avg_reward = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(episodes[window - 1:], moving_avg_reward, color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Steps
    axes[0, 1].plot(episodes, episode_steps, alpha=0.2, color='green')
    if len(episode_steps) >= window:
        moving_avg_steps = np.convolve(episode_steps, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(episodes[window - 1:], moving_avg_steps, color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Number of Steps')
    axes[0, 1].set_title('Episode Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    axes[1, 0].plot(episodes, actor_losses, alpha=0.2, color='purple', label='Actor')
    axes[1, 0].plot(episodes, critic_losses, alpha=0.2, color='brown', label='Critic')
    if len(actor_losses) >= window:
        moving_avg_actor = np.convolve(actor_losses, np.ones(window) / window, mode='valid')
        moving_avg_critic = np.convolve(critic_losses, np.ones(window) / window, mode='valid')
        axes[1, 0].plot(episodes[window - 1:], moving_avg_actor, color='purple', linewidth=2)
        axes[1, 0].plot(episodes[window - 1:], moving_avg_critic, color='brown', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 1].plot(episodes, entropies, alpha=0.2, color='teal')
    if len(entropies) >= window:
        moving_avg_entropy = np.convolve(entropies, np.ones(window) / window, mode='valid')
        axes[1, 1].plot(episodes[window - 1:], moving_avg_entropy, color='darkblue', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'all_metrics.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    reinforce_train_cat_monsters()