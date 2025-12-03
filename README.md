## Reinforcement Learning for Cache Replacement

This project implements and evaluates two policy–gradient–based RL algorithms  
on a custom **cache replacement** environment:

- **REINFORCE with Baseline** (episodic Monte Carlo policy gradient)
- **One-Step Actor–Critic** (episodic TD(0)-based policy gradient)

The objective is to learn a cache-eviction strategy that maximizes the cache-hit rate
over a sequence of page requests.

---

## 1. Environment: Cache Replacement MDP

### 1.1 Informal Description

We simulate a cache of fixed capacity $C$ that serves requests for $N$ distinct pages.

At each time step $t$:

1. A page $p_t$ is requested.
2. If $p_t$ is already in the cache: **cache hit**.
3. If $p_t$ is not in the cache:
   - If cache is not full: insert $p_t$.
   - If cache is full: the agent chooses *which page to evict*.
4. The agent receives reward based on hit/miss.

The goal is to maximize the long-term average reward (equivalently, hit rate).

---

### 1.2 MDP Formulation

#### States

The state at time $t$ is defined as:

$$s_t = (\mathbf{c}_t, \mathbf{p}_t)$$

where

- $\mathbf{c}_t \in \{0,1\}^N$ encodes cache contents:
  $$c_t[i] = \mathbf{1}\{\text{page } i \text{ is in cache at time } t\}$$

- $\mathbf{p}_t \in \{0,1\}^N$ is one-hot for the current request:
  $$p_t[i] = \mathbf{1}\{\text{page } i \text{ is requested at time } t\}$$

The full state vector is of length $2N$.

#### Actions

The action space is

$$\mathcal{A} = \{0, 1, \dots, N-1\}$$

An action $a_t$ means:

$$a_t = \text{"Evict page } a_t \text{ if eviction is needed."}$$

If the request is a hit, the action is ignored (no-op).

#### Transition Dynamics

Given $(s_t, a_t)$:

1. Determine hit/miss for the request $p_t$.
2. If miss and cache is full, evict page $a_t$ (with fallback if invalid).
3. Insert $p_t$ if needed.
4. Sample next request $p_{t+1}$ from a fixed stationary distribution.
5. Construct next state:
   $$s_{t+1} = (\mathbf{c}_{t+1}, \mathbf{p}_{t+1})$$

#### Reward

$$r_t = \begin{cases}
+1, & \text{hit} \\
-1, & \text{miss}
\end{cases}$$

#### Episodes

An episode has a fixed horizon $T$.
The return is:

$$G = \sum_{t=0}^{T-1} r_t$$

A discount factor $\gamma \in (0,1]$ is used in learning.

---

## 2. Policy and Value Parameterization

### Policy Network

The stochastic policy is $\pi_\theta(a|s)$.

Given state $s \in \mathbb{R}^{2N}$, the network outputs logits for all actions and a softmax produces:

$$\pi_\theta(a|s) = \frac{\exp(z_a)}{\sum_{i=0}^{N-1} \exp(z_i)}$$

### Value Network

The critic approximates the state-value function:

$$V_w(s) \approx \mathbb{E}[G_t | s_t = s]$$

Both are MLPs trained with Adam.

---

## 3. Algorithms

We implement:

1. **REINFORCE with Baseline**
2. **One-Step Actor–Critic**

---

## 3.1 REINFORCE with Baseline

### Returns

For an episode:

$$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$$

### Advantage

$$A_t = G_t - V_w(s_t)$$

### Policy Gradient Update

$$\theta \leftarrow \theta + \alpha_\pi \, A_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

### Value Update

$$w \leftarrow w - \alpha_v \, \nabla_w \left(V_w(s_t) - G_t\right)^2$$

---

## 3.2 One-Step Actor–Critic

### TD Error

For transition $(s_t, a_t, r_t, s_{t+1})$:

$$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$$

If the episode ends at $t+1$:

$$\delta_t = r_t - V_w(s_t)$$

### Critic Update

$$w \leftarrow w + \alpha_v \, \delta_t \, \nabla_w V_w(s_t)$$

### Actor Update

$$\theta \leftarrow \theta + \alpha_\pi \, \delta_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

---

## 4. Project Structure

```
.
├── env_cache.py          # CacheEnv implementation
├── models.py             # PolicyNet and ValueNet
├── reinforce.py          # REINFORCE with baseline
├── actor_critic.py       # One-step Actor-Critic
├── train_reinforce.py    # Training script
├── train_actor_critic.py # Training script
├── utils.py              # Plotting, seeding, etc.
└── plots/                # Saved plots
```
