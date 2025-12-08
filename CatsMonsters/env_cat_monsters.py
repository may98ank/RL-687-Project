import numpy as np
import random

class CatMonstersEnv:

    
    def __init__(self, discount_factor=0.925, seed=None):
        """
        Initialize the Cat and Monsters environment.
        
        Args:
            discount_factor: Discount factor for rewards
            seed: Random seed for reproducibility
        """
        self.states = [(r, c) for r in range(5) for c in range(5)]
        self.actions = ['AU', 'AD', 'AL', 'AR']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}
        
        self.forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
        self.monsters = [(0, 3), (4, 1)]
        self.food = (4, 4)
        self.discount_factor = discount_factor
        
        # State and action dimensions for neural networks
        self.state_dim = 25  # 5x5 grid, one-hot encoded
        self.action_dim = 4  # 4 actions
        
        # Current state
        self.current_state = None
        self.t = 0
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _state_to_onehot(self, state):
        if state == 's_infinity':
            return np.zeros(25)
        r, c = state
        idx = r * 5 + c
        onehot = np.zeros(25)
        onehot[idx] = 1.0
        return onehot
    
    def _is_valid_state(self, state):
        if state == 's_infinity':
            return True
        r, c = state
        return 0 <= r <= 4 and 0 <= c <= 4
    
    def _is_forbidden_furniture(self, state):
        return state in self.forbidden_furniture
    
    def _get_confused_right_direction(self, action):
        confused_right = {
            'AU': (0, 1),  
            'AD': (0, -1),  
            'AL': (-1, 0),  
            'AR': (1, 0)    
        }
        return confused_right[action]
    
    def _get_confused_left_direction(self, action):
        confused_left = {
            'AU': (0, -1),  
            'AD': (0, 1),   
            'AL': (1, 0),   
            'AR': (-1, 0)   
        }
        return confused_left[action]
    
    def _get_all_possible_next_states(self, state, action):
    
        r, c = state
        probabilities = {}
        
        directions = {
            'AU': (-1, 0),  
            'AD': (1, 0),   
            'AL': (0, -1),  
            'AR': (0, 1)    
        }
        
        intended_dr, intended_dc = directions[action]
        
        confused_right_dr, confused_right_dc = self._get_confused_right_direction(action)
        confused_left_dr, confused_left_dc = self._get_confused_left_direction(action)
        
        intended_pos = (r + intended_dr, c + intended_dc)
        confused_right_pos = (r + confused_right_dr, c + confused_right_dc)
        confused_left_pos = (r + confused_left_dr, c + confused_left_dc)
        
        if self._is_valid_state(intended_pos) and not self._is_forbidden_furniture(intended_pos):
            probabilities[intended_pos] = probabilities.get(intended_pos, 0) + 0.7
        else:
            probabilities[state] = probabilities.get(state, 0) + 0.7
        
        if self._is_valid_state(confused_right_pos) and not self._is_forbidden_furniture(confused_right_pos):
            probabilities[confused_right_pos] = probabilities.get(confused_right_pos, 0) + 0.12
        else:
            probabilities[state] = probabilities.get(state, 0) + 0.12
        
        if self._is_valid_state(confused_left_pos) and not self._is_forbidden_furniture(confused_left_pos):
            probabilities[confused_left_pos] = probabilities.get(confused_left_pos, 0) + 0.12
        else:
            probabilities[state] = probabilities.get(state, 0) + 0.12
        
        probabilities[state] = probabilities.get(state, 0) + 0.06
        
        return probabilities
    
    def _get_reward(self, state, action, next_state):
        """Get the reward for a given state, action, and next state."""
        if state == self.food:
            return 0.0
        elif next_state == self.food:
            return 10.0
        elif next_state in self.monsters:
            return -8.0
        return -0.05
    
    def _get_next_state(self, state, action):
        """
        Sample the next state given current state and action.
        Returns the next state (actual state or 'terminal' if food is reached).
        """
        # Handle terminal state: if current state is food, episode ends
        if state == self.food:
            return 'terminal'
        
        # Get all possible next states and their probabilities
        possible_states = self._get_all_possible_next_states(state, action)
        
        # Sample based on probabilities
        states = list(possible_states.keys())
        probabilities = list(possible_states.values())
        next_state = random.choices(states, weights=probabilities, k=1)[0]
        
        return next_state
    
    def reset(self):
       
        valid_states = [s for s in self.states if s not in self.forbidden_furniture and s != self.food]
        self.current_state = random.choice(valid_states)
        self.t = 0
        return self._state_to_onehot(self.current_state)
    
    def step(self, action_idx, verbose=False):
        
        action = self.idx_to_action[action_idx]
        
        next_state = self._get_next_state(self.current_state, action)
        
        reached_food = False
        if next_state == 'terminal':
            reward = 10.0  
            done = True
            reached_food = True
            self.current_state = None
        else:
            reward = self._get_reward(self.current_state, action, next_state)
            done = False
            self.current_state = next_state
        
        self.t += 1
        
        if done:
            state_vector = np.zeros(25)  
        else:
            state_vector = self._state_to_onehot(self.current_state)
        
        info = {
            'state': self.current_state,
            'action': action,
            'hit_monster': next_state in self.monsters if next_state != 'terminal' else False,
            'reached_food': reached_food
        }
        
        if verbose:
            print(f"Step {self.t}: State {self.current_state}, Action {action}, Next: {next_state}, Reward: {reward}, Done: {done}")
        
        return state_vector, reward, done, info
    
    def get_state(self):
        if self.current_state is None:
            return np.zeros(25)
        return self._state_to_onehot(self.current_state)
    
    def print_state(self):
        print(f"\n{'='*60}")
        print(f"Time Step: {self.t}")
        print(f"{'='*60}")
        if self.current_state is None:
            print("State: Terminal")
        else:
            r, c = self.current_state
            print(f"Current position: ({r}, {c})")
            print(f"State one-hot index: {r * 5 + c}")
            state_vec = self._state_to_onehot(self.current_state)
            print(f"State vector: {state_vec}")
        self.render()
    
    def render(self):
        """Print a visual representation of the grid."""
        grid = [['.' for _ in range(5)] for _ in range(5)]
        
        # Mark special cells
        grid[4][4] = 'F'  # Food
        grid[0][3] = 'M'  # Monster
        grid[4][1] = 'M'  # Monster
        for r, c in self.forbidden_furniture:
            grid[r][c] = 'X'  # Forbidden
        
        # Mark current position
        if self.current_state is not None:
            r, c = self.current_state
            grid[r][c] = 'C'  # Cat
        
        print("\nGrid:")
        for row in grid:
            print(' '.join(row))
        print()


# Test the environment
if __name__ == "__main__":
    env = CatMonstersEnv(seed=42)
    
    print("="*60)
    print("CAT AND MONSTERS ENVIRONMENT TEST")
    print("="*60)
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Actions: {env.actions}")
    print(f"Food location: {env.food}")
    print(f"Monster locations: {env.monsters}")
    print(f"Forbidden furniture: {env.forbidden_furniture}")
    print("="*60)
    
    # Test a few steps
    state = env.reset()
    print(f"\nInitial state (one-hot): {state}")
    env.render()
    
    total_reward = 0
    for step in range(20):
        action = random.randint(0, 3)
        next_state, reward, done, info = env.step(action, verbose=True)
        total_reward += reward
        env.render()
        
        if done:
            print(f"\nEpisode ended! Total reward: {total_reward}")
            break
        
        state = next_state
    
    print(f"\nFinal total reward: {total_reward}")
