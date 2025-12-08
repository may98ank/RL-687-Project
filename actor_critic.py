import numpy as np
import torch
from torch.distributions import Categorical


# ---------------------------------------------------------
# STATE NORMALIZATION (critical for stability)
# ---------------------------------------------------------
def normalize_state(s, env):
    x, x_dot, th, th_dot = s

    return np.array([
        x / env.x_threshold,
        x_dot / 10.0,
        th / env.theta_threshold_radians,
        th_dot / 10.0
    ], dtype=np.float32)


# ---------------------------------------------------------
# ONE-STEP ACTOR–CRITIC (TD(0))
# ---------------------------------------------------------
def train_actor_critic(
        env,
        policy_net,
        value_net,
        opt_actor,
        opt_critic,
        num_episodes=2000,
        gamma=0.99,
        entropy_coef=0.01,
        normalize=True,
        device="cpu",
        verbose=True
):
    """
    One-step Actor–Critic algorithm (episodic TD(0)).

    Logs returned:
        - rewards
        - lengths
        - actor_loss
        - critic_loss
        - td_error
    """
    policy_net.to(device)
    value_net.to(device)

    rewards_log = []
    lengths_log = []
    actor_loss_log = []
    critic_loss_log = []
    td_error_log = []

    for ep in range(num_episodes):

        s = env.reset()
        if normalize:
            s = normalize_state(s, env)
        s = torch.tensor(s, dtype=torch.float32, device=device)

        done = False
        ep_reward = 0
        ep_len = 0

        ep_actor_losses = []
        ep_critic_losses = []
        ep_deltas = []

        while not done:
            # --------------------------
            # 1. ACTOR: sample action
            # --------------------------
            logits = policy_net(s)
            dist = Categorical(logits=logits)
            a = dist.sample()
            log_prob = dist.log_prob(a)

            # --------------------------
            # 2. Step environment
            # --------------------------
            s2_np, r, done, info = env.step(a.item())
            if normalize:
                s2_np = normalize_state(s2_np, env)
            s2 = torch.tensor(s2_np, dtype=torch.float32, device=device)

            # --------------------------
            # 3. TD(0) ADVANTAGE
            # δ = r + γV(s') - V(s)
            # --------------------------
            V_s = value_net(s)
            
            if V_s.dim() > 0:
                V_s = V_s.squeeze()
            with torch.no_grad():
                V_s2 = value_net(s2) if not done else torch.tensor(0.0, device=device)
                if V_s2.dim() > 0:
                    V_s2 = V_s2.squeeze()

            td_target = r + gamma * V_s2
            delta = td_target - V_s

            # --------------------------
            # 4. Critic update: minimize δ^2
            # --------------------------
            critic_loss = delta.pow(2)
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

            # --------------------------
            # 5. Actor update: policy gradient with baseline
            # --------------------------
            entropy = dist.entropy()
            actor_loss = -log_prob * delta.detach() - entropy_coef * entropy
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            # --------------------------
            # Logging
            # --------------------------
            ep_reward += r
            ep_len += 1
            ep_actor_losses.append(actor_loss.item())
            ep_critic_losses.append(critic_loss.item())
            ep_deltas.append(delta.item())

            s = s2

        # Episode summary logs
        rewards_log.append(ep_reward)
        lengths_log.append(ep_len)
        actor_loss_log.append(float(np.mean(ep_actor_losses)))
        critic_loss_log.append(float(np.mean(ep_critic_losses)))
        td_error_log.append(float(np.mean(ep_deltas)))

        if verbose and (ep % 100 == 0 or ep == num_episodes - 1):
            print(
                f"[Episode {ep}] "
                f"Return={ep_reward:.1f}  Length={ep_len}  "
                f"A_Loss={actor_loss_log[-1]:.4f}  "
                f"C_Loss={critic_loss_log[-1]:.4f}  "
                f"TD={td_error_log[-1]:.4f}"
            )

    return {
        "rewards": rewards_log,
        "lengths": lengths_log,
        "actor_loss": actor_loss_log,
        "critic_loss": critic_loss_log,
        "td_error": td_error_log,
    }