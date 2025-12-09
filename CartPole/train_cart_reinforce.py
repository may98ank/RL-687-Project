import os
import argparse
import numpy as np
import torch
from torch.optim import Adam

from cartpole_env import CartPoleEnv
from models import PolicyNetwork, ValueNetwork
from reinforce import sample_episode, reinforce_update


 
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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CartPoleEnv(max_episode_steps=500, seed=args.seed)

    state_dim = 4
    action_dim = 2

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    # Optimizers
    opt_actor = Adam(policy_net.parameters(), lr=args.actor_lr)
    opt_critic = Adam(value_net.parameters(), lr=args.critic_lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    value_net.to(device)

    # Training logs
    episode_rewards = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    entropies = []

    print("="*60)
    print("TRAINING REINFORCE ON CARTPOLE")
    print("="*60)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Gamma: {args.gamma}")
    print(f"Entropy coefficient: {args.entropy_coef}")
    print(f"Device: {device}")
    print("="*60)

    # Train
    for ep in range(args.num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, args.gamma)
        
        actor_loss, critic_loss, entropy = reinforce_update(
            policy_net, value_net, opt_actor, opt_critic,
            states, actions, rewards, args.gamma, device,
            normalize_advantages=True,
            entropy_coef=args.entropy_coef,
            max_grad_norm=0.5
        )
        
        episode_rewards.append(stats['total_reward'])
        episode_lengths.append(stats['num_steps'])
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        entropies.append(entropy)
        
        # Print progress
        if (ep + 1) % 100 == 0 or ep == args.num_episodes - 1:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            print(
                f"[Episode {ep+1}/{args.num_episodes}] "
                f"Return={stats['total_reward']:.1f}  Length={stats['num_steps']}  "
                f"Avg Return (last 100)={avg_reward:.1f}  Avg Length (last 100)={avg_length:.1f}  "
                f"A_Loss={actor_loss:.4f}  C_Loss={critic_loss:.4f}  Entropy={entropy:.4f}"
            )

    # Create logs dictionary (same format as Actor-Critic for consistency)
    logs = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "actor_loss": actor_losses,
        "critic_loss": critic_losses,
        "td_error": [0.0] * args.num_episodes  # REINFORCE doesn't use TD error
    }

    # Save directories
    os.makedirs("plots/reinforce_cartpole/", exist_ok=True)
    os.makedirs("checkpoints/reinforce_cartpole/", exist_ok=True)

    # Save individual plots
    plot_curve(logs["rewards"], "plots/reinforce_cartpole/rewards.png", "Episode Rewards")
    plot_curve(logs["lengths"], "plots/reinforce_cartpole/lengths.png", "Episode Lengths")
    plot_curve(logs["actor_loss"], "plots/reinforce_cartpole/actor_loss.png", "Actor Loss")
    plot_curve(logs["critic_loss"], "plots/reinforce_cartpole/critic_loss.png", "Critic Loss")
    plot_curve(entropies, "plots/reinforce_cartpole/entropy.png", "Policy Entropy")
    
    # Create combined grid plot
    import matplotlib.pyplot as plt
    episodes = np.arange(1, args.num_episodes + 1)
    window = 100
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top left: Episode Rewards
    axes[0, 0].plot(episodes, episode_rewards, alpha=0.2, color='blue')
    if len(episode_rewards) >= window:
        moving_avg_reward = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(episodes[window - 1:], moving_avg_reward, color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top right: Episode Lengths
    axes[0, 1].plot(episodes, episode_lengths, alpha=0.2, color='green')
    if len(episode_lengths) >= window:
        moving_avg_lengths = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(episodes[window - 1:], moving_avg_lengths, color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Number of Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom left: Training Losses
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
    
    # Bottom right: Policy Entropy
    axes[1, 1].plot(episodes, entropies, alpha=0.2, color='teal')
    if len(entropies) >= window:
        moving_avg_entropy = np.convolve(entropies, np.ones(window) / window, mode='valid')
        axes[1, 1].plot(episodes[window - 1:], moving_avg_entropy, color='darkblue', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("plots/reinforce_cartpole/all_metrics.png", dpi=150)
    plt.close(fig)

    # Save weights
    torch.save(policy_net.state_dict(), "checkpoints/reinforce_cartpole/policy_net_cartpole_reinforce.pth")
    torch.save(value_net.state_dict(), "checkpoints/reinforce_cartpole/value_net_cartpole_reinforce.pth")


if __name__ == "__main__":
    main()

