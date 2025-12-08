import numpy as np


class CartPoleEnv:
    """
    A complete implementation of the classic control CartPole environment.
    No Gym used — all physics and dynamics coded manually.

    State = [x, x_dot, theta, theta_dot]
        x          — cart position
        x_dot      — cart velocity
        theta      — pole angle (radians)
        theta_dot  — pole angular velocity

    Actions:
        0 => push cart to the left (-force_mag)
        1 => push cart to the right (+force_mag)

    Reward:
        +1 for every timestep the pole remains upright
    """

    def __init__(self, max_episode_steps=500, seed=None):
        # ----- Physics constants -----
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5                     # actual pole length is 1.0 m
        self.polemass_length = self.masspole * self.length

        self.force_mag = 10.0
        self.tau = 0.02                       # seconds per timestep

        # ----- Termination thresholds (same as classic CartPole) -----
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * (np.pi / 180)

        # ----- Episode length limit -----
        self.max_episode_steps = max_episode_steps
        self.steps_beyond_done = None
        self.step_count = 0

        # ----- Seeding -----
        self.np_random = np.random.RandomState(seed)

        self.state = None

    # --------------------------------------------------------
    # reset()
    # --------------------------------------------------------
    def reset(self):
        """
        Reset the environment by sampling a small random initial state.
        """
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.step_count = 0
        self.steps_beyond_done = None
        return self.state.astype(np.float32)

    # --------------------------------------------------------
    # step(action)
    # --------------------------------------------------------
    def step(self, action):
        """
        Apply physics for one timestep using Euler integration.
        Returns (next_state, reward, done, info)
        """
        assert action in [0, 1], "Action must be 0 (left) or 1 (right)."
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # ---- Physics equations from the original CartPole source ----
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        theta_acc = (self.gravity * sintheta - costheta * temp) / \
                    (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))

        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        # ---- Euler integration ----
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.step_count += 1

        # ---- Termination conditions ----
        terminated = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        truncated = self.step_count >= self.max_episode_steps
        done = terminated or truncated

        # ---- Reward ----
        reward = 1.0 if not terminated else 0.0

        return self.state.astype(np.float32), reward, done, {}

    def render(self):
        print(f"state={self.state}")

    def seed(self, seed):
        self.np_random.seed(seed)
