from cmath import pi
import numpy as np
import torch
from torch import nn

class ActorNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=0.0005):
        super(ActorNetwork, self).__init__()
        
        self.input = nn.Linear(n_states, 400)   # Input Layer                
        self.fc1 = nn.Linear(400, 300)       # Hidden Layer1        
        self.fc2 = nn.Linear(300, n_actions) # Hidden Layer2
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=400, eps=0.001, momentum=0.01, affine=False)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.input.weight.data = fanin_init(self.input.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w) 

    def forward(self, state):        
        out = self.input(state)       
        # out = self.bn1(out) 
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)

        return out

class CriticNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=0.00005):
        super(CriticNetwork, self).__init__()
        
        self.input = nn.Linear(n_states + n_actions, 400)
        self.fc1 = nn.Linear(400, 300)                
        self.fc2 = nn.Linear(300, 1)
        self.relu = nn.LeakyReLU()        


        self.init_weights(init_w)

    def init_weights(self, init_w):    
        self.input.weight.data = fanin_init(self.input.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        out = self.input(torch.cat([state, action], 1))         
        out = self.relu(out)        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)