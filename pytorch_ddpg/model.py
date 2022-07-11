from cmath import pi
import numpy as np
import torch
from torch import nn

class ActorNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        
        self.input = nn.Linear(n_states, 600)   # Input Layer                
        self.fc1 = nn.Linear(600, 300)       # Hidden Layer1        
        self.fc2 = nn.Linear(300, n_actions) # Hidden Layer2
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.input.weight.data = fanin_init(self.input.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w) 

    def forward(self, state):        
        out = self.input(state)        
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)

        return out

class CriticNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=3e-4):
        super(CriticNetwork, self).__init__()
        
        self.state_input = nn.Linear(n_states, 600)     # Input Layer of states
        self.action_input = nn.Linear(n_actions, 300)   # Input Layer of actions
        self.fc1 = nn.Linear(600, 300)            
        self.fc2 = nn.Linear(300, 150)      
        self.output = nn.Linear(150, 1)
        self.bn1 = nn.BatchNorm1d(num_features=600)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        self.bn3 = nn.BatchNorm1d(num_features=150)        
        self.relu = nn.LeakyReLU()        
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.state_input.weight.data = fanin_init(self.state_input.weight.data.size())
        self.action_input.weight.data = fanin_init(self.action_input.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        state_out = self.state_input(state)
        state_out = self.relu(state_out)
        action_out = self.action_input(action)
        action_out = self.relu(action_out)                
        state_out = self.bn1(state_out)
        state_out = self.fc1(state_out)
        state_out = self.relu(state_out)
        add = state_out + action_out
        add = self.bn2(add)
        out = self.fc2(add)
        out = self.relu(out)        
        out = self.bn3(out)
        out = self.output(out)

        return out

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)