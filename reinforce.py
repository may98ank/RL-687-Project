# Implementing the REINFORCE WITH BASELINE algorithm

import torch 
import numpy as np
from torch.distributions import Categorical
from models import PolicyNetwork, ValueNetwork
from models import PolicyNetwork, ValueNetwork
from env_cache import CacheEnv
import torch.nn.functional as F


def sample_episode(env: CacheEnv, policy_net: PolicyNetwork, device: torch.device, gamma: float = 0.99, verbose: bool = False) -> tuple:
    """
    Sample an episode using the current policy.
    
    Args:
        env: CacheEnv environment
        policy_net: Policy network
        device: torch device
        gamma: discount factor
        verbose: If True, print detailed episode information
    
    Returns:
        states, actions, rewards, and optionally additional info if verbose
    """
    states, actions, rewards = [], [], []
    requests, hits, cache_states = [], [], []
    
    state = env.reset()
    done = False
    
    if verbose:
        print("\n" + "="*60)
        print("EPISODE EXECUTION (REINFORCE)")
        print("="*60)
        print(f"🔵 INITIAL STATE:")
        env.print_state()
        print("\n" + "─"*60)

    step = 0
    while not done:
        # Store current request before step (since step updates it)
        current_req = env.current_request
        cache_before = env.cache.copy()
        
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_logits = policy_net(s_tensor)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        
        next_state, reward, done, info = env.step(action.item())
        
        states.append(s_tensor)
        actions.append(action) # action is a tensor of shape (1,)
        rewards.append(reward)
        requests.append(current_req)
        hits.append(info["hit"])
        cache_states.append(env.cache.copy())  # Preserve order - slots matter now
        
        if verbose:
            hit_str = "HIT ✓" if info["hit"] else "MISS ✗"
            action_prob = action_dist.probs[action.item()].item()
            action_val = action.item()
            print(f"\nStep {step}:")
            print(f"  Cache (before): {cache_before if cache_before else '[]'}")
            print(f"  Request:        Page {current_req}")
            print(f"  Action:         Evict cache slot {action_val}")
            if cache_before and action_val < len(cache_before):
                print(f"  Slot {action_val} contains: Page {cache_before[action_val]}")
            print(f"  Result:         {hit_str}")
            print(f"  Reward:         {reward:+d}")
            print(f"  Cache (after):  {env.cache if env.cache else '[]'}")
            print(f"  Action prob:    {action_prob:.4f}")
        
        state = next_state
        step += 1
    
    # Compute episode statistics
    num_hits = sum(1 for r in rewards if r == 1)
    num_misses = len(rewards) - num_hits
    hit_rate = num_hits / len(rewards) if len(rewards) > 0 else 0.0
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards) if rewards else 0.0
    
    #TODO: reformat printing statements 
    if verbose:
        print("\n" + "="*60)
        print("EPISODE SUMMARY")
        print("="*60)
        
        print(f"\n📊 Statistics:")
        print(f"  Total steps: {len(rewards)}")
        print(f"  Hits: {num_hits} ({hit_rate*100:.1f}%)")
        print(f"  Misses: {num_misses} ({(1-hit_rate)*100:.1f}%)")
        print(f"  Total reward: {total_reward}")
        print(f"  Average reward: {avg_reward:.2f}")
        
        print(f"\n📋 Step-by-step breakdown:")
        print(f"{'Step':<6} {'Request':<8} {'Action':<12} {'Hit?':<6} {'Reward':<8} {'Cache After'}")
        print("-" * 75)
        for i in range(len(rewards)):
            cache_after = cache_states[i] if i < len(cache_states) else []
            hit_str = "✓" if hits[i] else "✗"
            cache_str = str(cache_after) if cache_after else "[]"
            action_str = f"Slot {actions[i].item()}"
            print(f"{i+1:<6} {requests[i]:<8} {action_str:<12} {hit_str:<6} {rewards[i]:+8} {cache_str}")
        
        print("\n" + "="*60)
    
    episode_stats = {
        'total_reward': total_reward,
        'hit_rate': hit_rate,
        'num_hits': num_hits,
        'num_misses': num_misses,
        'avg_reward': avg_reward,
        'num_steps': len(rewards)
    }
    
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
                     device: torch.device = torch.device('cpu')) -> tuple:

    returns = torch.tensor(compute_returns(rewards, gamma),
                           dtype=torch.float32, device=device)
    states_tensor = torch.stack(states).to(device)
    actions_tensor = torch.tensor([a.item() for a in actions],
                                  dtype=torch.long, device=device)

    values = value_net(states_tensor)
    critic_loss = ((returns - values) ** 2).mean()

    logits = policy_net(states_tensor)
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen_log_probs = log_probs[torch.arange(len(actions)), actions_tensor]

    advantages = (returns - values).detach()
    actor_loss = -(chosen_log_probs * advantages).mean()

    optimiser_policy.zero_grad()
    actor_loss.backward()
    optimiser_policy.step()

    optimiser_value.zero_grad()
    critic_loss.backward()
    optimiser_value.step()

    return actor_loss.item(), critic_loss.item()

def train_reinforce(env: CacheEnv, policy_net: PolicyNetwork, value_net: ValueNetwork, optimiser_policy: torch.optim.Optimizer, 
                    optimiser_value: torch.optim.Optimizer, device: torch.device = torch.device('cpu'), num_episodes: int = 1000, 
                    gamma: float = 0.99, verbose: bool = False, freq_print: int = 10):
    
    episode_rewards = []
    episode_hit_rates = []
    for episode in range(num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, gamma, verbose=False)
        actor_loss, critic_loss = reinforce_update(policy_net, value_net, optimiser_policy, optimiser_value, states, actions, rewards, gamma, device)
        
        episode_rewards.append(stats['total_reward'])
        episode_hit_rates.append(stats['hit_rate'])
        
        if verbose and (episode + 1) % freq_print == 0:
            print(f"Episode {episode+1}/{num_episodes} - Total Reward: {stats['total_reward']}, Hit Rate: {stats['hit_rate']*100:.2f}%, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        
    return episode_rewards, episode_hit_rates
    

def smoke_test():
    device = "cuda"

    env = CacheEnv(num_pages=10, cache_size=3, episode_len=10)
    policy_net = PolicyNetwork(state_dim=20, action_dim=10).to(device)
    value_net = ValueNetwork(state_dim=20).to(device)

    optimiser_policy = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    optimiser_value = torch.optim.Adam(value_net.parameters(), lr=1e-4)

    state = env.reset()
    s_tensor = torch.tensor(state, dtype=torch.float32, device=device)

    logits = policy_net(s_tensor)
    vaalue = value_net(s_tensor)

    states, actions, rewards, stats = sample_episode(env, policy_net, device, verbose=True)

    rewards_hist, hitrates_hist = train_reinforce(env, policy_net, value_net, optimiser_policy, optimiser_value, device, num_episodes=100000, verbose=True, freq_print=100)
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

if __name__ == "__main__":
    test_sample_episode()

    print("="*60)

    smoke_test()