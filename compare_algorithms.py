"""
Compare REINFORCE and Actor-Critic algorithms on both CartPole and Cat and Monsters environments.
"""

import os
import argparse
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

from cartpole_env import CartPoleEnv
from env_cat_monsters import CatMonstersEnv
from models import PolicyNetwork, ValueNetwork
from reinforce import sample_episode, reinforce_update
from actor_critic import train_actor_critic, normalize_state


def train_reinforce_with_logs(env, env_name, num_episodes, gamma, entropy_coef, device='cpu', verbose=False):
    """Train REINFORCE and return logs in same format as Actor-Critic."""
    # Determine state and action dimensions based on environment
    if env_name == 'cartpole':
        state_dim = 4
        action_dim = 2
        normalize = True
    elif env_name == 'cats':
        state_dim = 25
        action_dim = 4
        normalize = False
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)
    
    opt_actor = Adam(policy_net.parameters(), lr=3e-4)
    opt_critic = Adam(value_net.parameters(), lr=1e-4)
    
    episode_rewards = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    
    for ep in range(num_episodes):
        # Sample episode with normalization for CartPole
        if normalize and env_name == 'cartpole':
            # For CartPole, we need to normalize states during sampling
            states, actions, rewards = [], [], []
            state = env.reset()
            state = normalize_state(state, env)
            done = False
            
            while not done:
                s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                from torch.distributions import Categorical
                action_logits = policy_net(s_tensor)
                action_dist = Categorical(logits=action_logits)
                action = action_dist.sample()
                
                next_state, reward, done, _ = env.step(action.item())
                if normalize:
                    next_state = normalize_state(next_state, env)
                
                states.append(s_tensor)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
            
            total_reward = sum(rewards)
            episode_stats = {
                'total_reward': total_reward,
                'avg_reward': np.mean(rewards) if rewards else 0.0,
                'num_steps': len(rewards)
            }
        else:
            states, actions, rewards, episode_stats = sample_episode(env, policy_net, device, gamma)
        
        actor_loss, critic_loss, _ = reinforce_update(
            policy_net, value_net, opt_actor, opt_critic,
            states, actions, rewards, gamma, device,
            normalize_advantages=True,
            entropy_coef=entropy_coef,
            max_grad_norm=0.5
        )
        
        episode_rewards.append(episode_stats['total_reward'])
        episode_lengths.append(episode_stats['num_steps'])
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        
        if verbose and (ep % 100 == 0 or ep == num_episodes - 1):
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"[REINFORCE-{env_name.upper()} Episode {ep}] Reward={episode_stats['total_reward']:.1f} "
                  f"Length={episode_stats['num_steps']} Avg Reward (last 100)={avg_reward:.2f}")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'actor_loss': actor_losses,
        'critic_loss': critic_losses,
        'td_error': [0.0] * num_episodes  # REINFORCE doesn't use TD error
    }


def plot_comparison(reinforce_logs, ac_logs, env_name, save_dir):
    """Plot side-by-side comparisons of REINFORCE and Actor-Critic for a given environment."""
    os.makedirs(save_dir, exist_ok=True)
    
    episodes_reinforce = np.arange(1, len(reinforce_logs['rewards']) + 1)
    episodes_ac = np.arange(1, len(ac_logs['rewards']) + 1)
    
    window = 50  # Moving average window
    
    # Combined Overlay Plot
    plt.figure(figsize=(15, 10))
    
    # Rewards
    plt.subplot(2, 2, 1)
    if len(reinforce_logs['rewards']) >= window:
        ma_reinforce = np.convolve(reinforce_logs['rewards'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_reinforce[window-1:], ma_reinforce, color='blue', linewidth=2, label='REINFORCE')
    if len(ac_logs['rewards']) >= window:
        ma_ac = np.convolve(ac_logs['rewards'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_ac[window-1:], ma_ac, color='red', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Average)')
    plt.title(f'{env_name.upper()}: Episode Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lengths
    plt.subplot(2, 2, 2)
    if len(reinforce_logs['lengths']) >= window:
        ma_reinforce = np.convolve(reinforce_logs['lengths'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_reinforce[window-1:], ma_reinforce, color='blue', linewidth=2, label='REINFORCE')
    if len(ac_logs['lengths']) >= window:
        ma_ac = np.convolve(ac_logs['lengths'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_ac[window-1:], ma_ac, color='red', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length (Moving Average)')
    plt.title(f'{env_name.upper()}: Episode Lengths Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Actor Loss
    plt.subplot(2, 2, 3)
    if len(reinforce_logs['actor_loss']) >= window:
        ma_reinforce = np.convolve(reinforce_logs['actor_loss'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_reinforce[window-1:], ma_reinforce, color='blue', linewidth=2, label='REINFORCE')
    if len(ac_logs['actor_loss']) >= window:
        ma_ac = np.convolve(ac_logs['actor_loss'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_ac[window-1:], ma_ac, color='red', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss (Moving Average)')
    plt.title(f'{env_name.upper()}: Actor Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Critic Loss
    plt.subplot(2, 2, 4)
    if len(reinforce_logs['critic_loss']) >= window:
        ma_reinforce = np.convolve(reinforce_logs['critic_loss'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_reinforce[window-1:], ma_reinforce, color='blue', linewidth=2, label='REINFORCE')
    if len(ac_logs['critic_loss']) >= window:
        ma_ac = np.convolve(ac_logs['critic_loss'], np.ones(window)/window, mode='valid')
        plt.plot(episodes_ac[window-1:], ma_ac, color='red', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss (Moving Average)')
    plt.title(f'{env_name.upper()}: Critic Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{env_name}_combined_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Comparison plots saved to {save_dir}")


def plot_cross_environment_comparison(all_logs, save_dir):
    """Plot comparisons across environments for each algorithm."""
    os.makedirs(save_dir, exist_ok=True)
    
    window = 50
    
    # REINFORCE: CartPole vs Cat and Monsters
    plt.figure(figsize=(15, 10))
    
    # Rewards - REINFORCE
    plt.subplot(2, 2, 1)
    if 'reinforce_cartpole' in all_logs:
        logs = all_logs['reinforce_cartpole']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='blue', linewidth=2, label='REINFORCE-CartPole')
    if 'reinforce_cats' in all_logs:
        logs = all_logs['reinforce_cats']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='cyan', linewidth=2, label='REINFORCE-Cats')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Average)')
    plt.title('REINFORCE: Cross-Environment Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rewards - Actor-Critic
    plt.subplot(2, 2, 2)
    if 'ac_cartpole' in all_logs:
        logs = all_logs['ac_cartpole']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='red', linewidth=2, label='Actor-Critic-CartPole')
    if 'ac_cats' in all_logs:
        logs = all_logs['ac_cats']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='orange', linewidth=2, label='Actor-Critic-Cats')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Average)')
    plt.title('Actor-Critic: Cross-Environment Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # All algorithms on CartPole
    plt.subplot(2, 2, 3)
    if 'reinforce_cartpole' in all_logs:
        logs = all_logs['reinforce_cartpole']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='blue', linewidth=2, label='REINFORCE')
    if 'ac_cartpole' in all_logs:
        logs = all_logs['ac_cartpole']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='red', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Average)')
    plt.title('CartPole: Algorithm Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # All algorithms on Cat and Monsters
    plt.subplot(2, 2, 4)
    if 'reinforce_cats' in all_logs:
        logs = all_logs['reinforce_cats']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='blue', linewidth=2, label='REINFORCE')
    if 'ac_cats' in all_logs:
        logs = all_logs['ac_cats']
        if len(logs['rewards']) >= window:
            episodes = np.arange(1, len(logs['rewards']) + 1)
            ma = np.convolve(logs['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color='red', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Average)')
    plt.title('Cat and Monsters: Algorithm Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_environment_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Cross-environment comparison plot saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare REINFORCE and Actor-Critic on both environments')
    parser.add_argument("--num_episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--gamma_cartpole", type=float, default=0.99, help="Discount factor for CartPole")
    parser.add_argument("--gamma_cats", type=float, default=0.925, help="Discount factor for Cat and Monsters")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="results/comparison/all", help="Directory to save results")
    parser.add_argument("--envs", type=str, nargs='+', default=['cartpole', 'cats'], 
                        help="Environments to compare: 'cartpole', 'cats', or both")
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    all_logs = {}
    
    print("="*60)
    print("COMPARING REINFORCE AND ACTOR-CRITIC")
    print("ON MULTIPLE ENVIRONMENTS")
    print("="*60)
    print(f"Environments: {args.envs}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Entropy coefficient: {args.entropy_coef}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    # Train on CartPole
    if 'cartpole' in args.envs:
        print("\n" + "="*60)
        print("CARTPOLE ENVIRONMENT")
        print("="*60)
        
        # REINFORCE on CartPole
        print("\n[1/4] Training REINFORCE on CartPole...")
        env_reinforce_cp = CartPoleEnv(max_episode_steps=500, seed=args.seed)
        reinforce_cp_logs = train_reinforce_with_logs(
            env_reinforce_cp,
            'cartpole',
            num_episodes=args.num_episodes,
            gamma=args.gamma_cartpole,
            entropy_coef=args.entropy_coef,
            device='cpu',
            verbose=True
        )
        all_logs['reinforce_cartpole'] = reinforce_cp_logs
        print(f"REINFORCE-CartPole training complete!")
        print(f"  Final average reward (last 100): {np.mean(reinforce_cp_logs['rewards'][-100:]):.2f}")
        
        # Actor-Critic on CartPole
        print("\n[2/4] Training Actor-Critic on CartPole...")
        env_ac_cp = CartPoleEnv(max_episode_steps=500, seed=args.seed)
        policy_net_ac_cp = PolicyNetwork(4, 2)
        value_net_ac_cp = ValueNetwork(4)
        opt_actor_ac_cp = Adam(policy_net_ac_cp.parameters(), lr=3e-4)
        opt_critic_ac_cp = Adam(value_net_ac_cp.parameters(), lr=1e-4)
        
        ac_cp_logs = train_actor_critic(
            env_ac_cp,
            policy_net_ac_cp,
            value_net_ac_cp,
            opt_actor_ac_cp,
            opt_critic_ac_cp,
            num_episodes=args.num_episodes,
            gamma=args.gamma_cartpole,
            entropy_coef=args.entropy_coef,
            normalize=True,
            device='cpu',
            verbose=True
        )
        all_logs['ac_cartpole'] = ac_cp_logs
        print(f"Actor-Critic-CartPole training complete!")
        print(f"  Final average reward (last 100): {np.mean(ac_cp_logs['rewards'][-100:]):.2f}")
        
        # Plot comparison for CartPole
        print("\nGenerating CartPole comparison plots...")
        plot_comparison(reinforce_cp_logs, ac_cp_logs, 'cartpole', args.save_dir)
    
    # Train on Cat and Monsters
    if 'cats' in args.envs:
        print("\n" + "="*60)
        print("CAT AND MONSTERS ENVIRONMENT")
        print("="*60)
        
        # REINFORCE on Cat and Monsters
        print("\n[3/4] Training REINFORCE on Cat and Monsters...")
        env_reinforce_cats = CatMonstersEnv(seed=args.seed)
        reinforce_cats_logs = train_reinforce_with_logs(
            env_reinforce_cats,
            'cats',
            num_episodes=args.num_episodes,
            gamma=args.gamma_cats,
            entropy_coef=args.entropy_coef,
            device='cpu',
            verbose=True
        )
        all_logs['reinforce_cats'] = reinforce_cats_logs
        print(f"REINFORCE-Cats training complete!")
        print(f"  Final average reward (last 100): {np.mean(reinforce_cats_logs['rewards'][-100:]):.2f}")
        
        # Actor-Critic on Cat and Monsters
        print("\n[4/4] Training Actor-Critic on Cat and Monsters...")
        env_ac_cats = CatMonstersEnv(seed=args.seed)
        policy_net_ac_cats = PolicyNetwork(25, 4)
        value_net_ac_cats = ValueNetwork(25)
        opt_actor_ac_cats = Adam(policy_net_ac_cats.parameters(), lr=3e-4)
        opt_critic_ac_cats = Adam(value_net_ac_cats.parameters(), lr=1e-4)
        
        ac_cats_logs = train_actor_critic(
            env_ac_cats,
            policy_net_ac_cats,
            value_net_ac_cats,
            opt_actor_ac_cats,
            opt_critic_ac_cats,
            num_episodes=args.num_episodes,
            gamma=args.gamma_cats,
            entropy_coef=args.entropy_coef,
            normalize=False,
            device='cpu',
            verbose=True
        )
        all_logs['ac_cats'] = ac_cats_logs
        print(f"Actor-Critic-Cats training complete!")
        print(f"  Final average reward (last 100): {np.mean(ac_cats_logs['rewards'][-100:]):.2f}")
        
        # Plot comparison for Cat and Monsters
        print("\nGenerating Cat and Monsters comparison plots...")
        plot_comparison(reinforce_cats_logs, ac_cats_logs, 'cats', args.save_dir)
    
    # Save all logs
    os.makedirs(args.save_dir, exist_ok=True)
    for key, logs in all_logs.items():
        np.save(os.path.join(args.save_dir, f'{key}_rewards.npy'), np.array(logs['rewards']))
        np.save(os.path.join(args.save_dir, f'{key}_lengths.npy'), np.array(logs['lengths']))
        np.save(os.path.join(args.save_dir, f'{key}_actor_loss.npy'), np.array(logs['actor_loss']))
        np.save(os.path.join(args.save_dir, f'{key}_critic_loss.npy'), np.array(logs['critic_loss']))
        if 'td_error' in logs:
            np.save(os.path.join(args.save_dir, f'{key}_td_error.npy'), np.array(logs['td_error']))
    
    # Generate cross-environment comparison
    if len(args.envs) == 2:
        print("\nGenerating cross-environment comparison plots...")
        plot_cross_environment_comparison(all_logs, args.save_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    
    for key, logs in all_logs.items():
        print(f"\n{key.upper()}:")
        print(f"  Average reward (all): {np.mean(logs['rewards']):.2f} ± {np.std(logs['rewards']):.2f}")
        print(f"  Average reward (last 100): {np.mean(logs['rewards'][-100:]):.2f}")
        print(f"  Average length (all): {np.mean(logs['lengths']):.2f} ± {np.std(logs['lengths']):.2f}")
        print(f"  Average length (last 100): {np.mean(logs['lengths'][-100:]):.2f}")
    
    print(f"\nResults saved to: {args.save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
