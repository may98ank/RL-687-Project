import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.optim import Adam

from env_cat_monsters import CatMonstersEnv
from models import QNetwork
from semi_gradient_n_step_sarsa import train_sarsa_n_step


def sarsa_train_cat_monsters():
    """Train semi-gradient n-step SARSA on Cat and Monsters environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Environment and model setup
    env = CatMonstersEnv(seed=42)
    state_dim = env.state_dim          # 25 (5x5 one-hot)
    action_dim = env.action_dim        # 4 actions: AU, AD, AL, AR

    q_net = QNetwork(state_dim=state_dim,
                     action_dim=action_dim,
                     hidden_dim=128).to(device)

    optimizer = Adam(q_net.parameters(), lr=3e-4)

    # Softmax policy over Q-values (Option A)
    def policy_fn(state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a state tensor (shape [state_dim]),
        return logits over actions.
        We use Q(s, ·) as logits -> softmax(Q) defines the policy.
        """
        with torch.no_grad():
            q_vals = q_net(state_tensor)
        return q_vals

    # Output directories
    plot_dir = "plots/cat_monsters_sarsa_n_step"
    ckpt_dir = "checkpoints/cats_sarsa_n_step"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training hyperparameters
    num_episodes = 3000
    n = 3          # n-step return
    gamma = 0.925  # same discount as your REINFORCE script

    # Train
    logs = train_sarsa_n_step(
        env=env,
        q_net=q_net,
        optimizer=optimizer,
        policy_fn=policy_fn,
        device=device,
        n=n,
        num_episodes=num_episodes,
        gamma=gamma,
        normalize_fn=None,
        verbose=True,
        print_every=100
    )

    episode_rewards = logs["rewards"]
    episode_steps = logs["lengths"]
    losses = logs["loss"]

    # Save Q-network
    q_path = os.path.join(ckpt_dir, "q_net_cat_monsters_sarsa_n_step.pth")
    torch.save(q_net.state_dict(), q_path)
    print("Saved Q-network to:", q_path)

    # Plotting
    episodes = np.arange(1, num_episodes + 1)
    window = 100

    # 1) Episode Reward
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, alpha=0.3, color="blue", label="Episode Reward")
    if len(episode_rewards) >= window:
        moving_avg_reward = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            episodes[window - 1:], moving_avg_reward,
            color="red", linewidth=2,
            label=f"Moving Average ({window} episodes)"
        )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward Over Training (n-step SARSA)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "episode_reward.png"), dpi=150)
    plt.close()

    # 2) Episode Steps
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_steps, alpha=0.3, color="green", label="Episode Steps")
    if len(episode_steps) >= window:
        moving_avg_steps = np.convolve(
            episode_steps, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            episodes[window - 1:], moving_avg_steps,
            color="orange", linewidth=2,
            label=f"Moving Average ({window} episodes)"
        )
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.title("Episode Steps Over Training (n-step SARSA)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "episode_steps.png"), dpi=150)
    plt.close()

    # 3) Loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses, alpha=0.3, color="purple", label="TD Loss")
    if len(losses) >= window:
        moving_avg_loss = np.convolve(
            losses, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            episodes[window - 1:], moving_avg_loss,
            color="purple", linewidth=2,
            label=f"TD Loss MA ({window})"
        )
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("TD Loss Over Training (n-step SARSA)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_loss.png"), dpi=150)
    plt.close()

    # 4) Combined metrics (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Reward
    axes[0, 0].plot(episodes, episode_rewards, alpha=0.2, color="blue")
    if len(episode_rewards) >= window:
        axes[0, 0].plot(
            episodes[window - 1:], moving_avg_reward,
            color="red", linewidth=2
        )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # Steps
    axes[0, 1].plot(episodes, episode_steps, alpha=0.2, color="green")
    if len(episode_steps) >= window:
        axes[0, 1].plot(
            episodes[window - 1:], moving_avg_steps,
            color="orange", linewidth=2
        )
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Number of Steps")
    axes[0, 1].set_title("Episode Steps")
    axes[0, 1].grid(True, alpha=0.3)

    # Loss
    axes[1, 0].plot(episodes, losses, alpha=0.2, color="purple")
    if len(losses) >= window:
        axes[1, 0].plot(
            episodes[window - 1:], moving_avg_loss,
            color="purple", linewidth=2
        )
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("TD Loss")
    axes[1, 0].set_title("TD Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Empty / placeholder or could reuse reward entropy etc.
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Reserved")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "all_metrics.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    sarsa_train_cat_monsters()
