import torch
import numpy as np
from models import PolicyNetwork, ValueNetwork
from env_cat_monsters import CatMonstersEnv
import matplotlib.pyplot as plt
import os

def sample_episode_greedy(env: CatMonstersEnv, policy_net: PolicyNetwork, device: torch.device) -> dict:
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_logits = policy_net(s_tensor)
        # greedy action
        action = torch.argmax(action_logits).item()
        next_state, reward, done, _ = env.step(action)
        states.append(s_tensor)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    total_reward = sum(rewards)
    avg_reward = np.mean(rewards) if rewards else 0.0
    
    episode_stats = {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'num_steps': len(rewards)
    }
    
    return states, actions, rewards,episode_stats
def eval_cat_monsters_env(env: CatMonstersEnv, policy_net: PolicyNetwork, device: torch.device, num_episodes: int=1000) -> dict:
   
    episode_rewards = []
    episode_steps = []

    for _ in range(num_episodes):
        _, _, _, stats = sample_episode_greedy(env, policy_net, device)
        episode_rewards.append(stats['total_reward'])
        episode_steps.append(stats['num_steps'])
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,

    }


def load_policy_and_value_networks():
    policy_net = PolicyNetwork(state_dim=25, action_dim=4, hidden_dim=128)
    value_net = ValueNetwork(state_dim=25, hidden_dim=128)
    policy_net.load_state_dict(torch.load("checkpoints/cats_reinforce/policy_net_cat_monsters_reinforce.pth"))
    value_net.load_state_dict(torch.load("checkpoints/cats_reinforce/value_net_cat_monsters_reinforce.pth"))
    return policy_net, value_net


def test_policy_and_value_networks():
    policy_net, value_net = load_policy_and_value_networks()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    value_net.to(device)
    env = CatMonstersEnv(seed=42)
    os.makedirs("eval_results/eval_reinforce", exist_ok=True)
    
    eval_results = eval_cat_monsters_env(env, policy_net, device, num_episodes=1000)
    
    print(f"\nEvaluation Results (over {len(eval_results['episode_rewards'])} episodes):")
    print(f"  Average Reward: {eval_results['avg_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"  Average Steps: {eval_results['avg_steps']:.4f} ± {eval_results['std_steps']:.4f}")
    print(f"  Min Reward: {np.min(eval_results['episode_rewards']):.4f}")
    print(f"  Max Reward: {np.max(eval_results['episode_rewards']):.4f}")
    print(f"  Min Steps: {np.min(eval_results['episode_steps']):.0f}")
    print(f"  Max Steps: {np.max(eval_results['episode_steps']):.0f}")
    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(eval_results['episode_rewards'], alpha=0.6, color='blue')
    plt.axhline(y=eval_results['avg_reward'], color='red', linestyle='--', linewidth=2, label=f'Mean: {eval_results["avg_reward"]:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(eval_results['episode_steps'], alpha=0.6, color='green')
    plt.axhline(y=eval_results['avg_steps'], color='orange', linestyle='--', linewidth=2, label=f'Mean: {eval_results["avg_steps"]:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(eval_results['episode_rewards'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=eval_results['avg_reward'], color='red', linestyle='--', linewidth=2, label=f'Mean: {eval_results["avg_reward"]:.2f}')
    plt.axvline(x=eval_results['avg_reward'] + eval_results['std_reward'], color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 Std: {eval_results["std_reward"]:.2f}')
    plt.axvline(x=eval_results['avg_reward'] - eval_results['std_reward'], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution (REINFORCE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(eval_results['episode_steps'], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(x=eval_results['avg_steps'], color='orange', linestyle='--', linewidth=2, label=f'Mean: {eval_results["avg_steps"]:.2f}')
    plt.axvline(x=eval_results['avg_steps'] + eval_results['std_steps'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 Std: {eval_results["std_steps"]:.2f}')
    plt.axvline(x=eval_results['avg_steps'] - eval_results['std_steps'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.title('Steps Distribution (REINFORCE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("eval_results/eval_reinforce/eval_results_reinforce.png", dpi=150)
    plt.close()
    
    return eval_results


if __name__ == "__main__":
    test_policy_and_value_networks()