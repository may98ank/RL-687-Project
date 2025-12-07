import torch
from torch.distributions import Categorical
import torch.nn.functional as F


# ============================================================
#   GENERIC REINFORCE WITH BASELINE (ENV-AGNOSTIC)
# ============================================================

def sample_episode(env, policy_net, device, normalize_fn=None):
    """
    Samples a full episode from any environment.
    Returns lists of (states, actions, rewards).
    """
    states, actions, rewards = [], [], []

    state = env.reset()
    if normalize_fn is not None:
        state = normalize_fn(state)

    done = False

    while not done:
        s = torch.tensor(state, dtype=torch.float32, device=device)
        logits = policy_net(s)
        dist = Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.item())

        if normalize_fn is not None:
            next_state = normalize_fn(next_state)

        states.append(s)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    return states, actions, rewards


def compute_returns(rewards, gamma):
    """Monte Carlo return calculation."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def reinforce_update(policy_net, value_net, opt_policy, opt_value,
                     states, actions, rewards, gamma, device):
    """
    Performs one REINFORCE-with-baseline update.
    Returns actor_loss, critic_loss.
    """
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    states_tensor = torch.stack(states).to(device)
    actions_tensor = torch.tensor([a.item() for a in actions],
                                  dtype=torch.long, device=device)

    # Critic update: fit V(s) to returns
    values = value_net(states_tensor)
    critic_loss = F.mse_loss(values, returns)

    # Actor update
    logits = policy_net(states_tensor)
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions_tensor)

    advantages = (returns - values).detach()
    actor_loss = -(log_probs * advantages).mean()

    opt_policy.zero_grad()
    actor_loss.backward()
    opt_policy.step()

    opt_value.zero_grad()
    critic_loss.backward()
    opt_value.step()

    return actor_loss.item(), critic_loss.item()


def train_reinforce(env,
                    policy_net,
                    value_net,
                    opt_policy,
                    opt_value,
                    device,
                    num_episodes=2000,
                    gamma=0.99,
                    normalize_fn=None,
                    verbose=True,
                    print_every=100):
    """
    Generic REINFORCE-with-baseline training loop.
    Works for ANY environment.
    """

    rewards_log = []
    lengths_log = []
    actor_loss_log = []
    critic_loss_log = []
    steps_log = []
    for ep in range(num_episodes):

        states, actions, rewards = sample_episode(
            env, policy_net, device, normalize_fn
        )

        actor_loss, critic_loss = reinforce_update(
            policy_net, value_net, opt_policy, opt_value,
            states, actions, rewards, gamma, device
        )

        rewards_log.append(sum(rewards))
        lengths_log.append(len(rewards))
        actor_loss_log.append(actor_loss)
        critic_loss_log.append(critic_loss)
        steps_log.append(len(rewards))

        if verbose and (ep % print_every == 0 or ep == num_episodes - 1):
            print(
                f"[Episode {ep}] Reward={rewards_log[-1]:.1f}, "
                f"Len={lengths_log[-1]}, "
                f"A_Loss={actor_loss:.4f}, "
                f"C_Loss={critic_loss:.4f},"
                f"Steps={steps_log[-1]:.1f}"
            )

    return {
        "rewards": rewards_log,
        "lengths": lengths_log,
        "actor_loss": actor_loss_log,
        "critic_loss": critic_loss_log,
        "step": steps_log
    }
