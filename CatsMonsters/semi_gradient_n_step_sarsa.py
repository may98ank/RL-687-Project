import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def sample_episode_n_step(env, policy_fn, device, normalize_fn=None):
    states, actions, rewards = [], [], []

    state = env.reset()
    done = False

    while not done:
        if normalize_fn:
            state = normalize_fn(state, env)

        s = torch.tensor(state, dtype=torch.float32, device=device)

        logits = policy_fn(s)
        dist = Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())

        states.append(s)
        actions.append(action.item()) 
        rewards.append(reward)

        state = next_state

    return states, actions, rewards


def compute_n_step_targets(states, actions, rewards, q_net, n, gamma, device):
    T = len(rewards)
    targets = torch.zeros(T, dtype=torch.float32, device=device)

    for t in range(T):
        G = 0.0
        power = 1.0

        for k in range(n):
            if t + k < T:
                G += power * rewards[t + k]
                power *= gamma
            else:
                break

        if t + n < T:
            with torch.no_grad():
                s_next = states[t + n]
                a_next = actions[t + n]
                q_boot = q_net.q_value(s_next, a_next)
                G += power * q_boot

        targets[t] = G

    return targets


def sarsa_n_step_update(q_net, optimizer, states, actions, targets, device):

    states_tensor = torch.stack(states).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)

    q_vals = q_net(states_tensor)

    idx = torch.arange(len(actions_tensor), device=device)
    chosen_q = q_vals[idx, actions_tensor]

    loss = F.mse_loss(chosen_q, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_sarsa_n_step(env,
                       q_net,
                       optimizer,
                       policy_fn,
                       device,
                       n=3,
                       num_episodes=2000,
                       gamma=0.99,
                       normalize_fn=None,
                       verbose=True,
                       print_every=100):

    rewards_log = []
    lengths_log = []
    loss_log = []

    for ep in range(num_episodes):

        states, actions, rewards = sample_episode_n_step(
            env, policy_fn, device, normalize_fn
        )

        targets = compute_n_step_targets(
            states, actions, rewards, q_net, n, gamma, device
        )

        loss = sarsa_n_step_update(
            q_net, optimizer, states, actions, targets, device
        )

        rewards_log.append(sum(rewards))
        lengths_log.append(len(rewards))
        loss_log.append(loss)

        if verbose and (ep % print_every == 0 or ep == num_episodes - 1):
            print(
                f"[Ep {ep}] Reward={rewards_log[-1]:.1f}, "
                f"Len={lengths_log[-1]}, "
                f"Loss={loss:.4f}"
            )

    return {
        "rewards": rewards_log,
        "lengths": lengths_log,
        "loss": loss_log
    }
