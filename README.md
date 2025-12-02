## RL Project ## 
# Reinforcement Learning for Cache Replacement

This project implements and evaluates two policy-gradient–based RL algorithms  
on a custom **cache replacement** environment:

- **REINFORCE with Baseline** (episodic, Monte Carlo policy gradient)
- **One-Step Actor-Critic** (episodic, TD(0)-based policy gradient)

The goal is to learn a cache eviction policy that maximizes the cache hit rate
over a sequence of page requests.

---

## 1. Environment: Cache Replacement MDP

### 1.1 Informal Description

We simulate a cache of fixed capacity `C` that serves requests for `N` distinct pages.

At each time step:

1. A page `p_t` is requested.
2. If `p_t` is already in the cache → **cache hit**.
3. If `p_t` is not in the cache:
   - If the cache is not full → insert `p_t`.
   - If the cache is full → the agent must choose **which page to evict**, then insert `p_t`.
4. The agent receives a reward based on whether the request was a hit or miss.

The objective is to learn an eviction policy that maximizes long-term average reward
(equivalently, the cache hit rate).

---

### 1.2 MDP Formulation

We formulate this as a Markov Decision Process (MDP):

- **States**:  
  At time \( t \), the state is
  
 $$ s_t = (\mathbf{c}_t, \mathbf{p}_t) $$
  
  where:
  - \( \mathbf{c}_t \in \{0,1\}^N \) is a binary vector encoding which pages are in the cache:
    \[
    c_t[i] = 1 \iff \text{page } i \text{ is in the cache at time } t.
    \]
  - \( \mathbf{p}_t \in \{0,1\}^N \) is a one-hot vector for the current request:
    \[
    p_t[i] = 1 \iff \text{page } i \text{ is requested at time } t.
    \]

  In code, we represent the state as a concatenated vector of length `2 * num_pages`.

- **Actions**:  
  The action space is the set of page indices:
  \[
  \mathcal{A} = \{0, 1, \dots, N-1\}.
  \]
  At each time step, the agent chooses an action \( a_t \), interpreted as:
  > "Evict page \( a_t \) from the cache if eviction is required."

  - If the current request is a **hit**, the cache contents do not need to change and the
    environment can ignore the action (or treat it as a no-op).
  - If the request is a **miss** and the cache is full, the environment attempts to evict
    the page indicated by \( a_t \). If the chosen page is not in the cache, a fallback
    strategy is applied (e.g., evict a default or random page). This is documented in the
    environment implementation.

- **Transition Dynamics**:  
  Given state \( s_t = (\mathbf{c}_t, \mathbf{p}_t) \) and action \( a_t \):

  1. The environment computes whether the request is a hit or miss.
  2. If it is a miss and the cache is full, it evicts a page based on \( a_t \) and inserts
     the requested page.
  3. The next request \( p_{t+1} \) is sampled from a fixed distribution over pages
     (e.g., uniform or skewed with “hot” pages).
  4. The next state \( s_{t+1} = (\mathbf{c}_{t+1}, \mathbf{p}_{t+1}) \) is constructed.

  Request generation is assumed to be **stationary and independent** of past requests,
  which makes the process Markov with respect to the state definition above.

- **Reward Function**:  
  At each time step:

  - \( r_t = +1 \) for a cache hit
  - \( r_t = -1 \) for a cache miss

  This directly encourages the agent to increase the hit rate.

- **Episodes**:  
  An episode consists of a fixed number of requests \( T \) (e.g., `episode_len = 100`):

  - The environment starts with an empty cache (or some initial configuration).
  - At each step, a new request is generated and the agent chooses an action.
  - After \( T \) steps, the episode terminates.
  - The **return** is the sum of rewards over the episode.

- **Discount Factor**:
  A discount factor \( \gamma \in (0,1] \) is used in the RL updates (typical choice: `γ = 0.99`).

---

### 1.3 Environment Implementation (Code-Level)

The environment is implemented as a simple Python class:

- `CacheEnv(num_pages, cache_size, episode_len)`
- Key methods:
  - `reset() -> state`:  
    Initializes `t = 0`, clears the cache, samples the first request, and returns the initial state vector.
  - `step(action) -> (next_state, reward, done, info)`:
    - Applies cache hit/miss logic and eviction based on `action` when necessary.
    - Samples the next request.
    - Advances the time step and returns:
      - the next state vector,
      - the scalar reward,
      - a boolean `done` flag,
      - an `info` dict (e.g., `{"hit": True/False}` for diagnostics).

A random policy and simple heuristic baselines (e.g., random eviction, LRU) are provided
to sanity-check the environment.

---

## 2. Policy and Value Function Parameterization

We use small neural networks to represent the policy and value functions.

- **Policy Network** \( \pi_\theta(a|s) \):
  - Input: state vector \( s \in \mathbb{R}^{2N} \)
  - Output: logits over `num_pages` actions
  - Implementation: 1–2 fully connected layers with `tanh` nonlinearity and a final
    linear layer, followed by a softmax to obtain action probabilities.

- **Value Network** \( V_w(s) \):
  - Input: state vector \( s \)
  - Output: scalar estimate \( V_w(s) \)
  - Implementation: similar MLP structure with a final scalar output layer.

Both networks are trained using PyTorch and optimized with Adam.

---

## 3. Algorithms

We implement two RL algorithms **from scratch**:

1. **REINFORCE with Baseline** (episodic Monte Carlo policy gradient)
2. **One-Step Actor-Critic** (TD(0)-based policy gradient)

Both operate on **episodic tasks** and use the same environment.

---

### 3.1 REINFORCE with Baseline

#### Idea

REINFORCE is a Monte Carlo policy gradient method that updates the policy parameters
in the direction that increases the log-probability of actions weighted by the return.
To reduce variance, we subtract a learned baseline—here, the state-value function \( V_w(s) \).

#### Return & Advantage

For an episode with rewards \( r_0, \dots, r_{T-1} \):
\[
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
\]

The advantage at time \( t \) is approximated as:
\[
A_t \approx G_t - V_w(s_t)
\]

#### Policy Update

We parameterize \( \pi_\theta(a|s) \) with a neural network and update:
\[
\theta \leftarrow \theta + \alpha_\pi \, A_t \, \nabla_\theta \log \pi_\theta(a_t | s_t)
\]

In practice, we average this over all time steps in an episode.

#### Baseline (Value) Update

The baseline parameters \( w \) are updated to regress \( V_w(s_t) \) toward the Monte Carlo return:
\[
w \leftarrow w - \alpha_v \nabla_w (V_w(s_t) - G_t)^2
\]

#### Training Loop (High-Level)

For each episode:

1. Roll out an episode using the current policy:
   - collect states, actions, and rewards
2. Compute returns \( G_t \) for all time steps.
3. Compute advantages \( G_t - V_w(s_t) \).
4. Update the policy network using the policy gradient.
5. Update the value network using mean-squared error to the returns.

The algorithm is implemented in `reinforce.py` with:

- `sample_episode(env, policy_net)` to gather trajectories
- `reinforce_update(...)` to perform one gradient update for a single episode

---

### 3.2 One-Step Actor-Critic (Episodic)

#### Idea

Actor-Critic combines:

- A **critic** that learns \( V_w(s) \) using TD(0), and
- An **actor** that updates the policy parameters using the TD error as an estimate
  of the advantage.

This makes updates more incremental and usually more sample-efficient than
pure Monte Carlo methods like REINFORCE.

#### TD Error

Given a transition \((s_t, a_t, r_t, s_{t+1})\):
\[
\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)
\]
(If the episode terminates at \( t+1 \), the target is just \( r_t \), i.e. no bootstrap term.)

#### Critic Update

We update the value function parameters \( w \) by minimizing the squared TD error:
\[
w \leftarrow w + \alpha_v \, \delta_t \, \nabla_w V_w(s_t)
\]

#### Actor Update

The policy parameters \( \theta \) are updated in the direction:
\[
\theta \leftarrow \theta + \alpha_\pi \, \delta_t \, \nabla_\theta \log \pi_\theta(a_t | s_t)
\]

Here, the TD error \( \delta_t \) acts as a bootstrapped estimate of the advantage.

#### Training Loop (High-Level)

For each episode:

1. Reset the environment and start from the initial state.
2. For each time step until termination:
   - Sample action \( a_t \sim \pi_\theta(\cdot | s_t) \).
   - Apply action in the environment to get next state and reward.
   - Compute TD error \( \delta_t \).
   - Update:
     - the critic parameters \( w \) with a TD(0) step,
     - the actor parameters \( \theta \) with a gradient step using \( \delta_t \).
3. Record the total reward for the episode.

The algorithm is implemented in `actor_critic.py` with:

- `actor_critic_episode(...)` to run one episode and perform step-wise updates
- a training loop that calls this function for many episodes

---

## 4. Project Structure

A possible directory layout:

```text
.
├── env_cache.py          # CacheEnv implementation
├── models.py             # PolicyNet and ValueNet
├── reinforce.py          # REINFORCE with baseline
├── actor_critic.py       # One-step Actor-Critic
├── train_reinforce.py    # Training script for REINFORCE
├── train_actor_critic.py # Training script for Actor-Critic
├── utils.py              # Seeding, plotting, etc.
├── README.md             # This file
└── plots/                # Saved learning curve and evaluation plots
