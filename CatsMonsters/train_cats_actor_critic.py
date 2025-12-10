import os
import argparse
import numpy as np
import torch
from torch.optim import Adam
from env_cat_monsters import CatMonstersEnv
from models import PolicyNetwork, ValueNetwork
from actor_critic import train_actor_critic

def plot_curve(values, save_path, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(values, alpha=0.5, label="raw")
    if len(values) >= 50:
        kernel = np.ones(50)/50
        smoothed = np.convolve(values, kernel, mode="valid")
        plt.plot(range(49, 49+len(smoothed)), smoothed, linewidth=2, label="smoothed")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=3000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.925)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = CatMonstersEnv(seed=args.seed)

    state_dim = 25  
    action_dim = 4  

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    opt_actor = Adam(policy_net.parameters(), lr=args.actor_lr)
    opt_critic = Adam(value_net.parameters(), lr=args.critic_lr)

    logs = train_actor_critic(
        env,
        policy_net,
        value_net,
        opt_actor,
        opt_critic,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        normalize=False,  
        device="cpu",
        verbose=True
    )

    os.makedirs("plots/cats_monsters_actor_critic/", exist_ok=True)
    os.makedirs("checkpoints/cats_actor_critic/", exist_ok=True)
    
    plot_curve(logs["rewards"], "plots/cats_monsters_actor_critic/rewards.png", "Episode Rewards")
    plot_curve(logs["lengths"], "plots/cats_monsters_actor_critic/lengths.png", "Episode Lengths")
    plot_curve(logs["actor_loss"], "plots/cats_monsters_actor_critic/actor_loss.png", "Actor Loss")
    plot_curve(logs["critic_loss"], "plots/cats_monsters_actor_critic/critic_loss.png", "Critic Loss")
    plot_curve(logs["entropy"], "plots/cats_monsters_actor_critic/entropy.png", "Policy Entropy")
    
    import matplotlib.pyplot as plt
    episodes = np.arange(1, args.num_episodes + 1)
    window = 100
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(episodes, logs["rewards"], alpha=0.2, color='blue')
    if len(logs["rewards"]) >= window:
        moving_avg_reward = np.convolve(logs["rewards"], np.ones(window) / window, mode='valid')
        axes[0, 0].plot(episodes[window - 1:], moving_avg_reward, color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(episodes, logs["lengths"], alpha=0.2, color='green')
    if len(logs["lengths"]) >= window:
        moving_avg_lengths = np.convolve(logs["lengths"], np.ones(window) / window, mode='valid')
        axes[0, 1].plot(episodes[window - 1:], moving_avg_lengths, color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Number of Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(episodes, logs["actor_loss"], alpha=0.2, color='purple', label='Actor')
    axes[1, 0].plot(episodes, logs["critic_loss"], alpha=0.2, color='brown', label='Critic')
    if len(logs["actor_loss"]) >= window:
        moving_avg_actor = np.convolve(logs["actor_loss"], np.ones(window) / window, mode='valid')
        moving_avg_critic = np.convolve(logs["critic_loss"], np.ones(window) / window, mode='valid')
        axes[1, 0].plot(episodes[window - 1:], moving_avg_actor, color='purple', linewidth=2)
        axes[1, 0].plot(episodes[window - 1:], moving_avg_critic, color='brown', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(episodes, logs["entropy"], alpha=0.2, color='teal')
    if len(logs["entropy"]) >= window:
        moving_avg_entropy = np.convolve(logs["entropy"], np.ones(window) / window, mode='valid')
        axes[1, 1].plot(episodes[window - 1:], moving_avg_entropy, color='darkblue', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    
    fig.tight_layout()
    fig.savefig("plots/cats_monsters_actor_critic/all_metrics.png", dpi=150)
    plt.close(fig)

    torch.save(policy_net.state_dict(), "checkpoints/cats_actor_critic/policy_net_cat_monsters_actor_critic.pth")
    torch.save(value_net.state_dict(), "checkpoints/cats_actor_critic/value_net_cat_monsters_actor_critic.pth")

    print("\nTraining Complete!")

if __name__ == "__main__":
    main()

