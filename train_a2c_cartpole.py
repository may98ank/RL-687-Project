import os
import argparse
import numpy as np
import torch
from torch.optim import Adam

from cartpole_env import CartPoleEnv
from models import PolicyNetwork, ValueNetwork
from actor_critic import train_actor_critic


# ---------------------------------------------------------
# OPTIONAL: Simple matplotlib plotter
# ---------------------------------------------------------
def plot_curve(values, save_path, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(values, alpha=0.5, label="raw")
    if len(values) >= 50:
        # 50-step moving average
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


# ---------------------------------------------------------
# TRAIN SCRIPT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Environment
    env = CartPoleEnv(max_episode_steps=500, seed=args.seed)

    # Networks
    state_dim = 4
    action_dim = 2

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    # Optimizers
    opt_actor = Adam(policy_net.parameters(), lr=args.actor_lr)
    opt_critic = Adam(value_net.parameters(), lr=args.critic_lr)

    # Train
    logs = train_actor_critic(
        env,
        policy_net,
        value_net,
        opt_actor,
        opt_critic,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        normalize=True,
        device="cpu",
        verbose=True
    )

    # Save directories
    os.makedirs("results/logs/", exist_ok=True)
    os.makedirs("results/plots/", exist_ok=True)
    os.makedirs("checkpoints/", exist_ok=True)

    # Save numpy logs
    np.save("results/logs/rewards.npy", np.array(logs["rewards"]))
    np.save("results/logs/lengths.npy", np.array(logs["lengths"]))
    np.save("results/logs/actor_loss.npy", np.array(logs["actor_loss"]))
    np.save("results/logs/critic_loss.npy", np.array(logs["critic_loss"]))
    np.save("results/logs/td_error.npy", np.array(logs["td_error"]))

    # Save plots
    plot_curve(logs["rewards"], "results/plots/rewards.png", "Episode Rewards")
    plot_curve(logs["lengths"], "results/plots/lengths.png", "Episode Lengths")
    plot_curve(logs["actor_loss"], "results/plots/actor_loss.png", "Actor Loss")
    plot_curve(logs["critic_loss"], "results/plots/critic_loss.png", "Critic Loss")
    plot_curve(logs["td_error"], "results/plots/td_error.png", "TD Error")

    # Save weights
    torch.save(policy_net.state_dict(), "checkpoints/policy_final.pth")
    torch.save(value_net.state_dict(), "checkpoints/value_final.pth")

    print("\nTraining Complete!")
    print("Logs saved in results/logs/")
    print("Plots saved in results/plots/")
    print("Checkpoints saved in checkpoints/")
    print("\nNow run evaluation or inspect logs for report.")


if __name__ == "__main__":
    main()
