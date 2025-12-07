import torch.nn as nn


# ---------------------------------------------------------
# POLICY NETWORK (π(a|s; θ))
# ---------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    Outputs logits over actions.
    Architecture kept simple for Cart-Pole:
      state_dim → 128 → 128 → action_dim
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)  # logits (NOT softmax)


# ---------------------------------------------------------
# VALUE NETWORK (V(s; w))
# ---------------------------------------------------------
class ValueNetwork(nn.Module):
    """
    Outputs a scalar state-value V(s).
    """
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
