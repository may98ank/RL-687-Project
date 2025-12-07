# Implementing the REINFORCE WITH BASELINE algorithm

import torch 
import numpy as np
from torch.distributions import Categorical
from models import PolicyNetwork, ValueNetwork
from env_cache import CacheEnv
from env_cat_monsters import CatMonstersEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def sample_episode(env, policy_net: PolicyNetwork, device: torch.device, gamma: float = 0.99, verbose: bool = False) -> tuple:
    """
    Sample an episode using the current policy.
    
    Args:
        env: Environment (CacheEnv or CatMonstersEnv)
        policy_net: Policy network
        device: torch device
        gamma: discount factor
        verbose: If True, print detailed episode information
    
    Returns:
        states, actions, rewards, and episode statistics
    """
    states, actions, rewards = [], [], []
    
    # Check if it's a cache environment or cat-monsters environment
    is_cache_env = hasattr(env, 'current_request')
    
    if is_cache_env:
        requests, hits, cache_states = [], [], []
    else:
        positions, reached_food, hit_monster = [], [], []
    
    state = env.reset()
    done = False

    if verbose:
        print("\n" + "="*60)
        print("EPISODE EXECUTION (REINFORCE)")
        print("="*60)
        print(f"🔵 INITIAL STATE:")
        if hasattr(env, 'print_state'):
            env.print_state()
        print("\n" + "─"*60)

    step = 0
    while not done:
        # Store environment-specific state before step
        if is_cache_env:
            current_req = env.current_request
            cache_before = env.cache.copy()
        else:
            pos_before = env.current_state
        
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_logits = policy_net(s_tensor)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        
        next_state, reward, done, info = env.step(action.item())
        
        states.append(s_tensor)
        actions.append(action)
        rewards.append(reward)
        
        if is_cache_env:
            requests.append(current_req)
            hits.append(info.get("hit", False))
            cache_states.append(env.cache.copy() if hasattr(env, 'cache') else [])
        else:
            positions.append(env.current_state if env.current_state is not None else 'terminal')
            reached_food.append(info.get("reached_food", False))
            hit_monster.append(info.get("hit_monster", False))
        
        if verbose:
            action_prob = action_dist.probs[action.item()].item()
            action_val = action.item()
            print(f"\nStep {step}:")
            if is_cache_env:
                hit_str = "HIT ✓" if info.get("hit", False) else "MISS ✗"
                print(f"  Cache (before): {cache_before if cache_before else '[]'}")
                print(f"  Request:        Page {current_req}")
                print(f"  Action:         Evict cache slot {action_val}")
                if cache_before and action_val < len(cache_before):
                    print(f"  Slot {action_val} contains: Page {cache_before[action_val]}")
                print(f"  Result:         {hit_str}")
                print(f"  Cache (after):  {env.cache if env.cache else '[]'}")
            else:
                action_names = ['AU', 'AD', 'AL', 'AR']
                print(f"  Position (before): {pos_before}")
                print(f"  Action:            {action_names[action_val]}")
                print(f"  Position (after):  {env.current_state if env.current_state is not None else 'terminal'}")
                if info.get("reached_food", False):
                    print(f"  Result:            Reached food! 🎉")
                elif info.get("hit_monster", False):
                    print(f"  Result:            Hit monster! 👹")
            print(f"  Reward:         {reward:+6.2f}")
            print(f"  Action prob:    {action_prob:.4f}")
        
        state = next_state
        step += 1
    
    # Compute episode statistics
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards) if rewards else 0.0
    
    if is_cache_env:
        num_hits = sum(1 for r in rewards if r == 1)
        num_misses = len(rewards) - num_hits
        hit_rate = num_hits / len(rewards) if len(rewards) > 0 else 0.0
        episode_stats = {
            'total_reward': total_reward,
            'hit_rate': hit_rate,
            'num_hits': num_hits,
            'num_misses': num_misses,
            'avg_reward': avg_reward,
            'num_steps': len(rewards)
        }
    else:
        episode_stats = {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'num_steps': len(rewards)
        }
    
    if verbose:
        print("\n" + "="*60)
        print("EPISODE SUMMARY")
        print("="*60)
        print(f"\n📊 Statistics:")
        print(f"  Total steps: {len(rewards)}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward: {avg_reward:.2f}")
        if is_cache_env:
            print(f"  Hits: {num_hits} ({hit_rate*100:.1f}%)")
            print(f"  Misses: {num_misses} ({(1-hit_rate)*100:.1f}%)")
        else:
            print(f"  Reached food: Yes (episode always terminates at food)")
        print("\n" + "="*60)
    
    return states, actions, rewards, episode_stats

def compute_returns(rewards: list, gamma: float = 0.99) -> list:
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def reinforce_update(policy_net: PolicyNetwork, value_net: ValueNetwork, optimiser_policy: torch.optim.Optimizer, 
                     optimiser_value: torch.optim.Optimizer, states: list, actions: list, rewards: list, gamma: float = 0.99, 
                     device: torch.device = torch.device('cpu'), normalize_advantages: bool = True, 
                     entropy_coef: float = 0.01, max_grad_norm: float = 0.5) -> tuple:
    """
    Perform one REINFORCE update for a single episode.
    
    Args:
        policy_net: Policy network
        value_net: Value network (baseline)
        optimiser_policy: Optimizer for policy network
        optimiser_value: Optimizer for value network
        states: List of state tensors
        actions: List of action tensors
        rewards: List of rewards
        gamma: Discount factor
        device: torch device
        normalize_advantages: Whether to normalize advantages
        entropy_coef: Coefficient for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        actor_loss, critic_loss, entropy
    """
    returns = torch.tensor(compute_returns(rewards, gamma),
                           dtype=torch.float32, device=device)
    states_tensor = torch.stack(states).to(device)
    actions_tensor = torch.tensor([a.item() for a in actions],
                                  dtype=torch.long, device=device)

    # Value network update
    values = value_net(states_tensor)
    # Ensure proper shape handling
    if len(values.shape) == 0:
        values = values.unsqueeze(0)
    elif len(values.shape) > 1:
        values = values.squeeze()
    critic_loss = F.mse_loss(values, returns)

    # Policy network update
    logits = policy_net(states_tensor)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    chosen_log_probs = log_probs[torch.arange(len(actions_tensor), device=device), actions_tensor]

    # Compute advantages
    advantages = (returns - values.detach())
    
    # Normalize advantages to reduce variance (critical for stable learning)
    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Policy loss with entropy bonus for exploration
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    actor_loss = -(chosen_log_probs * advantages).mean() - entropy_coef * entropy

    # Update policy network with gradient clipping
    optimiser_policy.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    optimiser_policy.step()

    # Update value network with gradient clipping
    optimiser_value.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
    optimiser_value.step()

    return actor_loss.item(), critic_loss.item(), entropy.item()

def train_reinforce(env: CacheEnv, policy_net: PolicyNetwork, value_net: ValueNetwork, optimiser_policy: torch.optim.Optimizer, 
                    optimiser_value: torch.optim.Optimizer, device: torch.device = torch.device('cpu'), num_episodes: int = 1000, 
                    gamma: float = 0.99, verbose: bool = False, freq_print: int = 10, normalize_advantages: bool = True,
                    entropy_coef: float = 0.01, max_grad_norm: float = 0.5):
    
    episode_rewards = []
    episode_hit_rates = []
    for episode in range(num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, gamma, verbose=False)
        actor_loss, critic_loss, entropy = reinforce_update(policy_net, value_net, optimiser_policy, optimiser_value, 
                                                               states, actions, rewards, gamma, device, normalize_advantages,
                                                               entropy_coef, max_grad_norm)
        
        episode_rewards.append(stats['total_reward'])
        episode_hit_rates.append(stats['hit_rate'])
        
        if verbose and (episode + 1) % freq_print == 0:
            avg_reward = np.mean(episode_rewards[-freq_print:]) if len(episode_rewards) >= freq_print else np.mean(episode_rewards)
            avg_hit_rate = np.mean(episode_hit_rates[-freq_print:]) if len(episode_hit_rates) >= freq_print else np.mean(episode_hit_rates)
            print(f"Episode {episode+1}/{num_episodes} - Avg Reward (last {freq_print}): {avg_reward:.2f}, "
                  f"Avg Hit Rate: {avg_hit_rate*100:.2f}%, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
                  f"Entropy: {entropy:.4f}")
        
    return episode_rewards, episode_hit_rates
    

def smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment parameters
    num_pages = 10
    cache_size = 3
    episode_len = 100  # Longer episodes for better learning signal
    
    env = CacheEnv(num_pages=num_pages, cache_size=cache_size, episode_len=episode_len)
    state_dim = 2 * num_pages
    action_dim = cache_size  # CRITICAL: action_dim must equal cache_size!
    
    policy_net = PolicyNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)
    value_net = ValueNetwork(state_dim=state_dim, hidden_dim=128).to(device)

    # Higher learning rates for faster learning
    optimiser_policy = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
    optimiser_value = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    # Test a single episode with verbose output
    states, actions, rewards, stats = sample_episode(env, policy_net, device, verbose=True)

    rewards_hist, hitrates_hist = train_reinforce(env, policy_net, value_net, optimiser_policy, optimiser_value, 
                                                    device, num_episodes=10000, verbose=True, freq_print=100,
                                                    normalize_advantages=True, entropy_coef=0.01, max_grad_norm=0.5)
    avg_hit = sum(hitrates_hist[-100:]) / 100
    avg_miss = 1 - avg_hit
    print(f"Average Hit Rate:  {avg_hit:.4f}")
    print(f"Average Miss Rate: {avg_miss:.4f}")

# test the sample_episode function
def test_sample_episode():
    print("="*60)
    print("REINFORCE ALGORITHM TEST")
    print("="*60)
    
    env = CacheEnv(num_pages=10, cache_size=3, episode_len=500)
    policy_net = PolicyNetwork(state_dim=20, action_dim=3)  # action_dim = cache_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    
    print(f"\nConfiguration:")
    print(f"  - Number of pages: {env.num_pages}")
    print(f"  - Cache size: {env.cache_size}")
    print(f"  - Episode length: {env.episode_len}")
    print(f"  - State dimension: 20 (2 * num_pages)")
    print(f"  - Action dimension: 3 (cache_size - cache slots 0 to {env.cache_size-1})")
    print(f"  - Device: {device}")
    print(f"  - Number of episodes: 100")
    
    # Run 100 episodes and collect statistics
    num_episodes = 100
    all_stats = []
    
    print(f"\n{'='*60}")
    print(f"Running {num_episodes} episodes...")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, verbose=True)
        all_stats.append(stats)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes...", end='\r')
    
    print(f"\n{'='*60}")
    print("AVERAGE STATISTICS ACROSS 100 EPISODES")
    print(f"{'='*60}")
    
    # Compute average statistics
    avg_total_reward = np.mean([s['total_reward'] for s in all_stats])
    avg_hit_rate = np.mean([s['hit_rate'] for s in all_stats])
    avg_num_hits = np.mean([s['num_hits'] for s in all_stats])
    avg_num_misses = np.mean([s['num_misses'] for s in all_stats])
    avg_episode_reward = np.mean([s['avg_reward'] for s in all_stats])
    avg_num_steps = np.mean([s['num_steps'] for s in all_stats])
    
    # Compute standard deviations
    std_total_reward = np.std([s['total_reward'] for s in all_stats])
    std_hit_rate = np.std([s['hit_rate'] for s in all_stats])
    std_avg_reward = np.std([s['avg_reward'] for s in all_stats])
    
    print(f"\n📊 Average Statistics:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Average steps per episode: {avg_num_steps:.2f}")
    print(f"  Average total reward: {avg_total_reward:.2f} ± {std_total_reward:.2f}")
    print(f"  Average reward per step: {avg_episode_reward:.2f} ± {std_avg_reward:.2f}")
    print(f"  Average hit rate: {avg_hit_rate*100:.2f}% ± {std_hit_rate*100:.2f}%")
    print(f"  Average hits per episode: {avg_num_hits:.2f}")
    print(f"  Average misses per episode: {avg_num_misses:.2f}")
    
    # Additional statistics
    best_episode = max(all_stats, key=lambda x: x['total_reward'])
    worst_episode = min(all_stats, key=lambda x: x['total_reward'])
    
    print(f"\n📈 Best Episode:")
    print(f"  Total reward: {best_episode['total_reward']}")
    print(f"  Hit rate: {best_episode['hit_rate']*100:.2f}%")
    print(f"  Hits: {best_episode['num_hits']}, Misses: {best_episode['num_misses']}")
    
    print(f"\n📉 Worst Episode:")
    print(f"  Total reward: {worst_episode['total_reward']}")
    print(f"  Hit rate: {worst_episode['hit_rate']*100:.2f}%")
    print(f"  Hits: {worst_episode['num_hits']}, Misses: {worst_episode['num_misses']}")
    
    print(f"\n{'='*60}")


def smoke_test_cat_monsters():
    """Smoke test for REINFORCE algorithm on Cat and Monsters environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("REINFORCE ALGORITHM TEST - CAT AND MONSTERS ENVIRONMENT")
    print("="*60)
    
    # Environment parameters
    env = CatMonstersEnv(seed=42)
    state_dim = 25  # 5x5 grid, one-hot encoded
    action_dim = 4  # 4 actions: AU, AD, AL, AR
    
    policy_net = PolicyNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)
    value_net = ValueNetwork(state_dim=state_dim, hidden_dim=128).to(device)
    
    optimiser_policy = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
    optimiser_value = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    print(f"\nConfiguration:")
    print(f"  - State dimension: {state_dim} (5x5 grid, one-hot)")
    print(f"  - Action dimension: {action_dim} (AU, AD, AL, AR)")
    print(f"  - Food location: {env.food}")
    print(f"  - Monster locations: {env.monsters}")
    print(f"  - Forbidden furniture: {env.forbidden_furniture}")
    print(f"  - Device: {device}")
    print("="*60)
    
    # Test a single episode with verbose output
    print("\n🔵 Testing a single episode:")
    states, actions, rewards, stats = sample_episode(env, policy_net, device, verbose=True)
    
    print(f"\n📊 Episode Statistics:")
    print(f"  Total steps: {stats['num_steps']}")
    print(f"  Total reward: {stats['total_reward']:.2f}")
    print(f"  Average reward per step: {stats['avg_reward']:.2f}")
    
    # Train the agent
    print(f"\n{'='*60}")
    print("TRAINING REINFORCE ON CAT AND MONSTERS ENVIRONMENT")
    print(f"{'='*60}")
    
    # Setup plot directory
    plot_dir = "plots/cat_monsters_reinforce"
    os.makedirs(plot_dir, exist_ok=True)
    
    episode_rewards = []
    episode_steps = []
    actor_losses = []
    critic_losses = []
    entropies = []
    
    num_episodes = 4000
    freq_print = 100
    
    for episode in range(num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, verbose=False)
        actor_loss, critic_loss, entropy = reinforce_update(policy_net, value_net, optimiser_policy, optimiser_value, 
                                                               states, actions, rewards, gamma=0.925, device=device, 
                                                               normalize_advantages=True, entropy_coef=0.01, max_grad_norm=0.5)
        
        episode_rewards.append(stats['total_reward'])
        episode_steps.append(stats['num_steps'])
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        entropies.append(entropy)
        
        # Print progress every freq_print episodes
        if (episode + 1) % freq_print == 0:
            avg_reward = np.mean(episode_rewards[-freq_print:])
            avg_steps = np.mean(episode_steps[-freq_print:])
            avg_actor_loss = np.mean(actor_losses[-freq_print:])
            avg_critic_loss = np.mean(critic_losses[-freq_print:])
            avg_entropy = np.mean(entropies[-freq_print:])
            
            print(f"Episode {episode+1}/{num_episodes} - Avg Reward (last {freq_print}): {avg_reward:.2f}, "
                  f"Avg Steps: {avg_steps:.1f}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
    
    # Save the policy and value networks
    torch.save(policy_net.state_dict(), "policy_net_cat_monsters.pth")
    torch.save(value_net.state_dict(), "value_net_cat_monsters.pth")
    
    # Plot learning curves
    episodes = np.arange(1, num_episodes + 1)
    
    # Plot 1: Episode Reward
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    # Moving average
    window = 100
    if len(episode_rewards) >= window:
        moving_avg_reward = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_reward, color='red', linewidth=2, label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'episode_reward.png'), dpi=150)
    plt.close()
    
    # Plot 2: Episode Steps
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_steps, alpha=0.3, color='green', label='Episode Steps')
    # Moving average
    if len(episode_steps) >= window:
        moving_avg_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_steps, color='orange', linewidth=2, label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Episode Steps Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'episode_steps.png'), dpi=150)
    plt.close()
    
    # Plot 3: Losses
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, actor_losses, alpha=0.3, color='purple', label='Actor Loss')
    plt.plot(episodes, critic_losses, alpha=0.3, color='brown', label='Critic Loss')
    # Moving averages
    if len(actor_losses) >= window:
        moving_avg_actor = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
        moving_avg_critic = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_actor, color='purple', linewidth=2, label=f'Actor Loss MA ({window})')
        plt.plot(episodes[window-1:], moving_avg_critic, color='brown', linewidth=2, label=f'Critic Loss MA ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_losses.png'), dpi=150)
    plt.close()
    
    # Plot 4: Entropy
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, entropies, alpha=0.3, color='teal', label='Policy Entropy')
    # Moving average
    if len(entropies) >= window:
        moving_avg_entropy = np.convolve(entropies, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg_entropy, color='darkblue', linewidth=2, label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'policy_entropy.png'), dpi=150)
    plt.close()
    
    # Plot 5: Combined metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward
    axes[0, 0].plot(episodes, episode_rewards, alpha=0.2, color='blue')
    if len(episode_rewards) >= window:
        moving_avg_reward = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg_reward, color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Steps
    axes[0, 1].plot(episodes, episode_steps, alpha=0.2, color='green')
    if len(episode_steps) >= window:
        moving_avg_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], moving_avg_steps, color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Number of Steps')
    axes[0, 1].set_title('Episode Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    axes[1, 0].plot(episodes, actor_losses, alpha=0.2, color='purple', label='Actor')
    axes[1, 0].plot(episodes, critic_losses, alpha=0.2, color='brown', label='Critic')
    if len(actor_losses) >= window:
        moving_avg_actor = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
        moving_avg_critic = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(episodes[window-1:], moving_avg_actor, color='purple', linewidth=2)
        axes[1, 0].plot(episodes[window-1:], moving_avg_critic, color='brown', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 1].plot(episodes, entropies, alpha=0.2, color='teal')
    if len(entropies) >= window:
        moving_avg_entropy = np.convolve(entropies, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(episodes[window-1:], moving_avg_entropy, color='darkblue', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'all_metrics.png'), dpi=150)
    plt.close()
    
    print(f"\n✅ Plots saved to: {plot_dir}")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    
    final_avg_reward = np.mean(episode_rewards[-100:])
    final_avg_steps = np.mean(episode_steps[-100:])
    best_reward = max(episode_rewards)
    worst_reward = min(episode_rewards)
    best_steps = min(episode_steps)
    worst_steps = max(episode_steps)
    
    print(f"\n📊 Last 100 Episodes:")
    print(f"  Average reward: {final_avg_reward:.2f}")
    print(f"  Average steps: {final_avg_steps:.1f}")
    print(f"  Best episode reward: {best_reward:.2f}")
    print(f"  Worst episode reward: {worst_reward:.2f}")
    print(f"  Best episode steps: {best_steps}")
    print(f"  Worst episode steps: {worst_steps}")
    
    print(f"\n📈 Overall Training:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Overall average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Overall average steps: {np.mean(episode_steps):.2f}")
    
    print(f"\n{'='*60}")

def load_policy_and_value_networks():
    policy_net = PolicyNetwork(state_dim=25, action_dim=4, hidden_dim=128)
    value_net = ValueNetwork(state_dim=25, hidden_dim=128)
    policy_net.load_state_dict(torch.load("policy_net_cat_monsters.pth"))
    value_net.load_state_dict(torch.load("value_net_cat_monsters.pth"))
    return policy_net, value_net

def test_policy_and_value_networks():
    # test on 100 episodes with the policy and return the average reward
    policy_net, value_net = load_policy_and_value_networks()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    value_net.to(device)
    env = CatMonstersEnv(seed=42)
    steps = []
    for episode in range(100):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, verbose=False)
        avg_reward = np.mean(rewards)
        steps.append(stats['num_steps'])
    # return avg_reward
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {np.mean(steps):.2f}")

if __name__ == "__main__":
    # test_sample_episode()
    # smoke_test()  # Uncomment to test cache environment
    smoke_test_cat_monsters()  # Test cat and monsters environment
    test_policy_and_value_networks()