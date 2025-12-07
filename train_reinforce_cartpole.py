import os
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

from cartpole_env import CartPoleEnv
from models import PolicyNetwork, ValueNetwork
from reinforce_generic import train_reinforce


# -------------------------
#  NORMALIZATION
# -------------------------

def normalize_cartpole(state, env):
    x, x_dot, theta, theta_dot = state
    return np.array([
        x / env.x_threshold,
        x_dot / 10.0,
        theta / env.theta_threshold_radians,
        theta_dot / 10.0
    ], dtype=np.float32)


# -------------------------
#  EVALUATION
# -------------------------

def evaluate_policy(env, policy_net, device, episodes=200):
    rewards = []
    lengths = []

    for _ in range(episodes):
        s = env.reset()
        s = normalize_cartpole(s, env)
        s = torch.tensor(s, dtype=torch.float32, device=device)

        done = False
        ep_reward = 0
        ep_len = 0

        while not done:
            with torch.no_grad():
                logits = policy_net(s)
                action = torch.argmax(logits).item()

            next_s, r, done, info = env.step(action)
            next_s = normalize_cartpole(next_s, env)

            ep_reward += r
            ep_len += 1
            s = torch.tensor(next_s, dtype=torch.float32, device=device)

        rewards.append(ep_reward)
        lengths.append(ep_len)

    metrics = {
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards),
        "success_rate": np.mean(np.array(lengths) >= env.max_episode_steps)
    }
    return metrics


# -------------------------
#  PLOTTER
# -------------------------

def plot_curve(values, title, filename):
    plt.figure(figsize=(8, 4))
    plt.plot(values, alpha=0.4)
    if len(values) >= 50:
        smooth = np.convolve(values, np.ones(50)/50, mode="valid")
        plt.plot(range(49, 49+len(smooth)), smooth, linewidth=2)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


# -------------------------
#  MAIN TRAIN + EVAL
# -------------------------

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    env = CartPoleEnv(max_episode_steps=500, seed=0)

    policy_net = PolicyNetwork(4, 2).to(device)
    value_net = ValueNetwork(4).to(device)

    opt_policy = Adam(policy_net.parameters(), lr=1e-3)
    opt_value = Adam(value_net.parameters(), lr=1e-3)

    logs = train_reinforce(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        opt_policy=opt_policy,
        opt_value=opt_value,
        device=device,
        num_episodes=2000,
        gamma=0.99,
        normalize_fn=lambda s: normalize_cartpole(s, env),
        verbose=True
    )

    os.makedirs("results_reinforce_cartpole/plots", exist_ok=True)

    plot_curve(logs["rewards"], "Rewards", "results_reinforce_cartpole/plots/rewards.png")
    plot_curve(logs["lengths"], "Episode Length", "results_reinforce_cartpole/plots/lengths.png")
    plot_curve(logs["actor_loss"], "Actor Loss", "results_reinforce_cartpole/plots/actor_loss.png")
    plot_curve(logs["critic_loss"], "Critic Loss", "results_reinforce_cartpole/plots/critic_loss.png")

    print("\nTraining finished.\nEvaluating...")

    metrics = evaluate_policy(
        CartPoleEnv(max_episode_steps=500, seed=123),
        policy_net,
        device
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k:15s}: {v}")


if __name__ == "__main__":
    main()
