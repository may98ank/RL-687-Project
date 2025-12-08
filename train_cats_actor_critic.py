import os
import argparse
import numpy as np
import torch
from torch.optim import Adam

from env_cat_monsters import CatMonstersEnv
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
    parser.add_argument("--num_episodes", type=int, default=3000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.925)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Environment
    env = CatMonstersEnv(seed=args.seed)

    # Networks
    state_dim = 25  # 5x5 grid, one-hot encoded
    action_dim = 4  # 4 actions: AU, AD, AL, AR

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
        normalize=False,  # One-hot vectors don't need normalization
        device="cpu",
        verbose=True
    )

    # Save directories
    os.makedirs("results/logs/cats_actor_critic/", exist_ok=True)
    os.makedirs("results/plots/cats_actor_critic/", exist_ok=True)
    os.makedirs("checkpoints/cats_actor_critic/", exist_ok=True)

    # Save numpy logs
    np.save("results/logs/cats_actor_critic/rewards.npy", np.array(logs["rewards"]))
    np.save("results/logs/cats_actor_critic/lengths.npy", np.array(logs["lengths"]))
    np.save("results/logs/cats_actor_critic/actor_loss.npy", np.array(logs["actor_loss"]))
    np.save("results/logs/cats_actor_critic/critic_loss.npy", np.array(logs["critic_loss"]))
    np.save("results/logs/cats_actor_critic/td_error.npy", np.array(logs["td_error"]))

    # Save plots
    plot_curve(logs["rewards"], "results/plots/cats_actor_critic/rewards.png", "Episode Rewards")
    plot_curve(logs["lengths"], "results/plots/cats_actor_critic/lengths.png", "Episode Lengths")
    plot_curve(logs["actor_loss"], "results/plots/cats_actor_critic/actor_loss.png", "Actor Loss")
    plot_curve(logs["critic_loss"], "results/plots/cats_actor_critic/critic_loss.png", "Critic Loss")
    plot_curve(logs["td_error"], "results/plots/cats_actor_critic/td_error.png", "TD Error")

    # Save weights
    torch.save(policy_net.state_dict(), "checkpoints/cats_actor_critic/policy_final.pth")
    torch.save(value_net.state_dict(), "checkpoints/cats_actor_critic/value_final.pth")

    print("\nTraining Complete!")
    print("Logs saved in results/logs/cats_actor_critic/")
    print("Plots saved in results/plots/cats_actor_critic/")
    print("Checkpoints saved in checkpoints/cats_actor_critic/")
    print("\nNow run evaluation or inspect logs for report.")


if __name__ == "__main__":
    main()

