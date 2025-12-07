import numpy as np
import torch
from torch.distributions import Categorical

from cartpole_env import CartPoleEnv
from models import PolicyNetwork


# ---------------------------------------------------------
# SAME NORMALIZATION USED IN TRAINING
# ---------------------------------------------------------
def normalize_state(s, env):
    x, x_dot, th, th_dot = s
    return np.array([
        x / env.x_threshold,
        x_dot / 10.0,
        th / env.theta_threshold_radians,
        th_dot / 10.0,
    ], dtype=np.float32)


# ---------------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------------
def evaluate_policy(
        env,
        policy_net,
        num_episodes=100,
        normalize=True,
        device="cpu",
        render=False
):
    rewards = []
    lengths = []
    success_count = 0

    policy_net.to(device)
    policy_net.eval()

    for ep in range(num_episodes):
        s = env.reset()
        if normalize:
            s = normalize_state(s, env)
        s = torch.tensor(s, dtype=torch.float32, device=device)

        done = False
        ep_reward = 0
        ep_len = 0

        while not done:
            if render:
                env.render()

            # Policy forward
            with torch.no_grad():
                logits = policy_net(s)
                dist = Categorical(logits=logits)
                a = dist.sample()

            # Step environment
            s2_np, r, done, info = env.step(a.item())
            if normalize:
                s2_np = normalize_state(s2_np, env)
            s2 = torch.tensor(s2_np, dtype=torch.float32, device=device)

            ep_reward += r
            ep_len += 1
            s = s2

        rewards.append(ep_reward)
        lengths.append(ep_len)

        # count success: reaching max steps
        if ep_len >= env.max_episode_steps:
            success_count += 1

    return {
        "num_episodes": num_episodes,
        "episode_rewards": np.array(rewards),
        "episode_lengths": np.array(lengths),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(success_count / num_episodes),
        "success_count": success_count,
    }


# ---------------------------------------------------------
# FUNCTION TO PRINT RESULTS
# ---------------------------------------------------------
def print_evaluation_results(metrics):
    print("\n==================== CARTPOLE EVALUATION ====================")
    print(f" Episodes:          {metrics['num_episodes']}")
    print(f" Mean reward:       {metrics['mean_reward']:.2f}")
    print(f" Min reward:        {metrics['min_reward']:.2f}")
    print(f" Max reward:        {metrics['max_reward']:.2f}")
    print(f" Mean length:       {metrics['mean_length']:.2f}")
    print(f" Success episodes:  {metrics['success_count']}")
    print(f" Success rate:      {metrics['success_rate']*100:.2f}%")
    print("=============================================================\n")


# ---------------------------------------------------------
# OPTIONAL: SCRIPT MODE
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--policy_path", type=str, default="checkpoints/policy_final.pth")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Create env
    env = CartPoleEnv(max_episode_steps=500)

    # Load policy
    policy = PolicyNetwork(state_dim=4, action_dim=2)
    policy.load_state_dict(torch.load(args.policy_path, map_location="cpu"))
    policy.eval()

    # Evaluate
    metrics = evaluate_policy(
        env,
        policy,
        num_episodes=args.episodes,
        normalize=True,
        device="cpu",
        render=args.render
    )

    print_evaluation_results(metrics)
