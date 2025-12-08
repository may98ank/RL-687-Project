import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from env_cat_monsters import CatMonstersEnv
from models import QNetwork  


def eval_cat_monsters_env_sarsa(env: CatMonstersEnv,
                                q_net: QNetwork,
                                device: torch.device,
                                num_episodes: int = 1000) -> dict:
    """
    Evaluate a trained n-step SARSA Q-network.
    Uses a greedy policy: a = argmax_a Q(s, a)
    """

    episode_rewards = []
    episode_steps = []

    for _ in range(num_episodes):

        state = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0

        while not done:

            s_tensor = torch.tensor(state, dtype=torch.float32, device=device)

            with torch.no_grad():
                q_vals = q_net(s_tensor.unsqueeze(0)).squeeze(0)
                action = torch.argmax(q_vals).item()

            next_state, reward, done, _ = env.step(action)

            ep_reward += reward
            ep_len += 1
            state = next_state

        episode_rewards.append(ep_reward)
        episode_steps.append(ep_len)

    return {
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_steps": float(np.mean(episode_steps)),
        "std_steps": float(np.std(episode_steps)),
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
    }


def load_q_network():
    """
    Load trained SARSA n-step Q-network.
    Assumes checkpoint is stored in: checkpoints/cats_sarsa_n/q_net.pth
    Modify as needed.
    """

    checkpoint_dir = "checkpoints/cats_sarsa_n_step"
    q_path = os.path.join(checkpoint_dir, "q_net_cat_monsters_sarsa_n_step.pth")

    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Q-network checkpoint not found at {q_path}")

    q_net = QNetwork(state_dim=25, action_dim=4, hidden_dim=128)
    q_net.load_state_dict(torch.load(q_path, map_location="cpu"))
    return q_net


def test_q_network():
    """Evaluate a trained SARSA n-step Q-network and generate plots."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = load_q_network()
    q_net.to(device)

    env = CatMonstersEnv(seed=42)

    eval_results = eval_cat_monsters_env_sarsa(env, q_net, device, num_episodes=500)

    print("\nSARSA n-step Evaluation Results:")
    print(f"  Average Reward: {eval_results['avg_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"  Average Steps: {eval_results['avg_steps']:.4f} ± {eval_results['std_steps']:.4f}")
    print(f"  Min Reward: {np.min(eval_results['episode_rewards']):.4f}")
    print(f"  Max Reward: {np.max(eval_results['episode_rewards']):.4f}")
    print(f"  Min Steps: {np.min(eval_results['episode_steps'])}")
    print(f"  Max Steps: {np.max(eval_results['episode_steps'])}")

    os.makedirs("eval_results/eval_sarsa_n_step", exist_ok=True)

    plt.figure(figsize=(12, 12))

    # Plot 1: Episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(eval_results["episode_rewards"], alpha=0.6, color="blue")
    plt.axhline(eval_results["avg_reward"], color="red", linestyle="--", label="Mean")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Episode Rewards (SARSA n-step)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Episode steps
    plt.subplot(2, 2, 2)
    plt.plot(eval_results["episode_steps"], alpha=0.6, color="green")
    plt.axhline(eval_results["avg_steps"], color="orange", linestyle="--", label="Mean")
    plt.ylabel("Steps")
    plt.xlabel("Episode")
    plt.title("Episode Steps (SARSA n-step)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Reward histogram
    plt.subplot(2, 2, 3)
    plt.hist(eval_results["episode_rewards"], bins=20, alpha=0.7, color="blue", edgecolor="black")
    plt.axvline(eval_results["avg_reward"], color="red", linestyle="--", label="Mean")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution (SARSA n-step)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Steps histogram
    plt.subplot(2, 2, 4)
    plt.hist(eval_results["episode_steps"], bins=20, alpha=0.7, color="green", edgecolor="black")
    plt.axvline(eval_results["avg_steps"], color="orange", linestyle="--", label="Mean")
    plt.xlabel("Steps")
    plt.ylabel("Frequency")
    plt.title("Steps Distribution (SARSA n-step)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = "eval_results/eval_sarsa_n_step/eval_results_sarsa_n_step.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nEvaluation plot saved to: {plot_path}")

    return eval_results


if __name__ == "__main__":
    test_q_network()
