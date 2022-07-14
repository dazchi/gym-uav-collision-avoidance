import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(n_states + n_actions, 600)
        self.fc2 = nn.Linear(600, 300)
        self.q = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q = torch.relu(self.fc1(sa))
        q = torch.relu(self.fc2(q))
        q = torch.relu(self.q(q))
        
        return q

class ValueNetwork(nn.Module):
    def __init__(self, n_states):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(n_states, 600)
        self.fc2 = nn.Linear(600, 300)
        self.v = nn.Linear(300, 1)

    def forward(self, state):        
        v = torch.relu(self.fc1(state))
        v = torch.relu(self.fc2(v))
        v = torch.relu(self.q(v))
        
        return v

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions) -> None:
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(n_states, 600)
        self.fc2 = nn.Linear(600, 300)
        self.mean = nn.Linear(300, n_actions)
        self.log_std = nn.Linear(300, n_actions)

    def forward(self, state):
        prob = torch.relu(self.fc1(state))
        prob = torch.relu(self.fc2(state))
        mean = self.mean(prob)
        log_std = self.log_std(prob)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        action = normal.rsample()