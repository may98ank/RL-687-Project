# RL-687 Project: Reinforcement Learning Algorithms Implementation

This repository contains implementations of three fundamental reinforcement learning algorithms applied to two different environments: CartPole and CatsMonsters. The project demonstrates policy gradient methods (REINFORCE, Actor-Critic) and value-based methods (N-step SARSA) using PyTorch.

## 📋 Table of Contents

- [Overview](#overview)
- [Environments](#environments)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## 🎯 Overview

This project implements and compares three reinforcement learning algorithms:

1. **REINFORCE** - Monte Carlo policy gradient method with value function baseline
2. **Actor-Critic** - Online policy gradient method with value function critic
3. **N-step SARSA** - Value-based method using n-step returns

Each algorithm is tested on two environments:
- **CartPole**: Classic control problem of balancing a pole on a cart
- **CatsMonsters**: Grid-world navigation task with stochastic transitions

## 🌍 Environments

### CartPole Environment

A custom implementation of the classic CartPole problem where an agent must balance a pole on a moving cart.

- **State Space**: 4D continuous (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 discrete actions (push left, push right)
- **Reward**: +1 for each step the pole remains balanced, 0 when episode terminates
- **Max Episode Steps**: 500
- **Termination**: Episode ends when pole falls beyond ±12° or cart moves beyond ±2.4 units

**Features**:
- State normalization for improved training stability
- Configurable episode length and random seed

### CatsMonsters Environment

A 5×5 grid-world environment where a cat must navigate to food while avoiding monsters.

- **State Space**: 25D one-hot encoded (5×5 grid positions)
- **Action Space**: 4 discrete actions (Up, Down, Left, Right)
- **Rewards**:
  - +10.0 for reaching food
  - -8.0 for hitting a monster
  - -0.05 for each step
- **Special Features**:
  - Stochastic transitions: Actions have 70% chance of intended direction, 12% each for confused directions, 6% for staying
  - Forbidden furniture cells that block movement
  - Episode terminates when food is reached

## 🤖 Algorithms

### 1. REINFORCE

Monte Carlo policy gradient algorithm that:
- Collects complete episodes before updating
- Uses Monte Carlo returns for policy gradient estimation
- Includes a value function baseline to reduce variance
- Supports advantage normalization and entropy regularization

**Key Features**:
- Episode-based updates
- Gradient clipping for stability
- Entropy bonus for exploration

### 2. Actor-Critic

Online policy gradient method that:
- Updates policy and value function after each step
- Uses TD error (δ) as advantage estimate
- Combines policy gradient with value function learning
- Includes entropy regularization

**Key Features**:
- Online learning (updates per step)
- Lower variance than REINFORCE
- Faster convergence

### 3. N-step SARSA

Value-based method that:
- Learns Q-function using n-step returns
- Uses ε-greedy policy derived from Q-values
- Combines n-step bootstrapping with function approximation

**Key Features**:
- Configurable n-step horizon
- Semi-gradient updates
- Can balance between Monte Carlo (large n) and TD(0) (n=1)

## 📁 Project Structure

```
RL-687-Project/
├── CartPole/
│   ├── cartpole_env.py              # Custom CartPole environment
│   ├── models.py                     # Neural network architectures
│   ├── reinforce.py                  # REINFORCE algorithm implementation
│   ├── actor_critic.py               # Actor-Critic algorithm implementation
│   ├── semi_gradient_n_step_sarsa.py # N-step SARSA implementation
│   ├── train_cart_reinforce.py      # Training script for REINFORCE
│   ├── train_cart_act_critic.py     # Training script for Actor-Critic
│   ├── train_cart_n_step_sarsa.py   # Training script for N-step SARSA
│   ├── eval_cart_reinforce.py       # Evaluation script for REINFORCE
│   ├── eval_cart_critic.py           # Evaluation script for Actor-Critic
│   └── eval_cart_n_step_sarsa.py     # Evaluation script for N-step SARSA
├── CatsMonsters/
│   ├── env_cat_monsters.py           # CatsMonsters grid-world environment
│   ├── models.py                     # Neural network architectures
│   ├── reinforce.py                  # REINFORCE algorithm implementation
│   ├── actor_critic.py               # Actor-Critic algorithm implementation
│   ├── semi_gradient_n_step_sarsa.py # N-step SARSA implementation
│   ├── train_reinforce_cats.py       # Training script for REINFORCE
│   ├── train_cats_actor_critic.py   # Training script for Actor-Critic
│   ├── train_n_step_sarsa_cats_vs_monster.py # Training script for N-step SARSA
│   ├── eval_reinforce_cats.py        # Evaluation script for REINFORCE
│   ├── eval_actor_critic_cats.py     # Evaluation script for Actor-Critic
│   └── eval_n_step_sarsa.py          # Evaluation script for N-step SARSA
└── README.md                         # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RL-687-Project
```

2. Install required packages:
```bash
pip install torch numpy matplotlib
```

## 🚀 Usage

### Training

#### CartPole - REINFORCE
```bash
cd CartPole
python train_cart_reinforce.py --num_episodes 3000 --gamma 0.99 --entropy_coef 0.01 --seed 0
```

#### CartPole - Actor-Critic
```bash
cd CartPole
python train_cart_act_critic.py --num_episodes 3000 --actor_lr 3e-4 --critic_lr 1e-4 --gamma 0.99 --entropy_coef 0.01 --seed 0
```

#### CartPole - N-step SARSA
```bash
cd CartPole
python train_cart_n_step_sarsa.py --num_episodes 3000 --n 3 --gamma 0.99 --lr 1e-3 --seed 0
```

#### CatsMonsters - REINFORCE
```bash
cd CatsMonsters
python train_reinforce_cats.py --num_episodes 3000 --gamma 0.925 --entropy_coef 0.01 --seed 0
```

#### CatsMonsters - Actor-Critic
```bash
cd CatsMonsters
python train_cats_actor_critic.py --num_episodes 3000 --actor_lr 3e-4 --critic_lr 1e-4 --gamma 0.925 --entropy_coef 0.01 --seed 0
```

#### CatsMonsters - N-step SARSA
```bash
cd CatsMonsters
python train_n_step_sarsa_cats_vs_monster.py --num_episodes 3000 --n 3 --gamma 0.925 --lr 1e-3 --seed 0
```

### Evaluation

After training, models are saved in `checkpoints/` directories. To evaluate:

#### CartPole
```bash
cd CartPole
python eval_cart_critic.py        # For Actor-Critic
python eval_cart_reinforce.py    # For REINFORCE
python eval_cart_n_step_sarsa.py # For N-step SARSA
```

#### CatsMonsters
```bash
cd CatsMonsters
python eval_actor_critic_cats.py  # For Actor-Critic
python eval_reinforce_cats.py     # For REINFORCE
python eval_n_step_sarsa.py       # For N-step SARSA
```

### Training Parameters

**Common Parameters**:
- `--num_episodes`: Number of training episodes (default: 3000)
- `--gamma`: Discount factor (default: 0.99 for CartPole, 0.925 for CatsMonsters)
- `--seed`: Random seed for reproducibility (default: 0)

**REINFORCE & Actor-Critic**:
- `--actor_lr`: Learning rate for policy network (default: 3e-4)
- `--critic_lr`: Learning rate for value network (default: 1e-4)
- `--entropy_coef`: Entropy regularization coefficient (default: 0.01)

**N-step SARSA**:
- `--n`: Number of steps for n-step returns (default: 3)
- `--lr`: Learning rate for Q-network (default: 1e-3)

## 📊 Results

Training generates the following outputs:

1. **Plots**: Saved in `plots/` directories
   - Episode rewards (raw and smoothed)
   - Episode lengths
   - Training losses (actor/critic/TD loss)
   - Policy entropy (for policy gradient methods)
   - TD errors (for Actor-Critic)

2. **Checkpoints**: Saved in `checkpoints/` directories
   - Trained model weights (`.pth` files)

3. **Evaluation Results**: 
   - Performance metrics (average reward, episode length)
   - Distribution plots
   - Evaluation visualizations

### Expected Performance

**CartPole**:
- Should achieve maximum episode length (500 steps) consistently
- Average reward should approach 500

**CatsMonsters**:
- Should learn to navigate to food while avoiding monsters
- Performance depends on stochastic transitions and exploration

## 🧠 Model Architectures

All neural networks use similar architectures:

- **PolicyNetwork**: 3-layer MLP with tanh activations
  - Input: state_dim → Hidden: 128 → Hidden: 128 → Output: action_dim
  
- **ValueNetwork**: 3-layer MLP with tanh activations
  - Input: state_dim → Hidden: 128 → Hidden: 128 → Output: 1
  
- **QNetwork**: 3-layer MLP with ReLU activations
  - Input: state_dim → Hidden: 128 → Hidden: 128 → Output: action_dim

## 📝 Notes

- State normalization is used for CartPole to improve training stability
- Gradient clipping (max_norm=0.5) is applied in REINFORCE
- Advantage normalization is used in REINFORCE for better stability
- All algorithms support GPU training (automatically detects CUDA availability)
- Evaluation uses greedy policy (argmax) for deterministic performance assessment

## 🔬 Algorithm Details

### REINFORCE
- Collects full episode trajectory
- Computes Monte Carlo returns: G_t = Σ(γ^k * r_{t+k})
- Updates policy: ∇θ J ≈ E[∇θ log π(a|s) * (G_t - V(s))]
- Updates value function: MSE(V(s), G_t)

### Actor-Critic
- Online updates after each step
- TD target: r + γ * V(s')
- TD error: δ = r + γ * V(s') - V(s)
- Policy update: ∇θ J ≈ E[∇θ log π(a|s) * δ]
- Value update: MSE(V(s), r + γ * V(s'))

### N-step SARSA
- N-step return: G_t = r_t + γ*r_{t+1} + ... + γ^n * Q(s_{t+n}, a_{t+n})
- Q-function update: MSE(Q(s_t, a_t), G_t)
- Policy: ε-greedy derived from Q-values

## 📄 License

This project is part of an academic course (RL-687) and is intended for educational purposes.

## 👥 Authors

RL-687 Course Project

---

For questions or issues, please refer to the course materials or contact the course instructor.
