from cmath import pi
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=0.0005):
        super(ActorNetwork, self).__init__()
        
        self.input = nn.Linear(n_states, 600)   # Input Layer                
        self.fc1 = nn.Linear(600, 300)       # Hidden Layer1        
        self.fc2 = nn.Linear(300, n_actions) # Hidden Layer2                        
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.input.weight.data = fanin_init(self.input.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w) 

    def forward(self, state):        
        out = F.leaky_relu(self.input(state))                
        out = F.leaky_relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))        

        return out

class CriticNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=0.00005):
        super(CriticNetwork, self).__init__()
        
        self.state_input = nn.Linear(n_states, 600)     # Input Layer of states
        self.action_input = nn.Linear(n_actions, 300)   # Input Layer of actions
        self.fc1 = nn.Linear(600, 300)            
        self.fc2 = nn.Linear(300, 150)    
        self.output = nn.Linear(150, 1)              
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.state_input.weight.data = fanin_init(self.state_input.weight.data.size())
        self.action_input.weight.data = fanin_init(self.action_input.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)       

    def forward(self, state, action):
        state_out = F.leaky_relu(self.state_input(state))        
        action_out = F.leaky_relu(self.action_input(action))        
        state_out = F.leaky_relu(self.fc1(state_out))    
        add = torch.add(state_out, action_out)                
        out = F.leaky_relu(self.fc2(add))        
        out = self.output(out)        

        return out

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)