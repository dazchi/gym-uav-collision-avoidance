import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from pytorch_ddpg.ou import OUActionNoise
# from pytorch_ddpg.model import ActorNetwork, CriticNetwork

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

from cmath import pi
import numpy as np
import torch
from torch import nn

class ActorNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=1e-4):
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
        out = self.relu(self.input(state))           
        out = self.relu(self.fc1(out))        
        out = self.tanh(self.fc2(out))        

        return out

class CriticNetwork(nn.Module):
    
    def __init__(self, n_states, n_actions, init_w=1e-3):
        super(CriticNetwork, self).__init__()
        
        self.state_input = nn.Linear(n_states + n_actions, 600)     # Input Layer of states        
        self.fc1 = nn.Linear(600, 300)            
        self.fc2 = nn.Linear(300, 1)              
        self.relu = nn.LeakyReLU()        
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.state_input.weight.data = fanin_init(self.state_input.weight.data.size())        
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)        

    def forward(self, state, action):
        out = self.relu(self.state_input(torch.concat([state, action], 1)))                       
        out = self.relu(self.fc1(out))           
        out = self.fc2(out)      
  
        return out


class TD3(object):
    def __init__(self, n_states, n_actions, buffer_size=1e5, batch_size=64, noise_std_dev=0.1, actor_lr=1e-4, critic_lr=1e-3, tau=0.001, gamma=0.99):        
        self.n_states = n_states
        self.n_actions = n_actions

        self.actor = ActorNetwork(self.n_states, self.n_actions)
        self.actor_target = ActorNetwork(self.n_states, self.n_actions)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, amsgrad=True)

        self.critic_1 = CriticNetwork(self.n_states, self.n_actions)
        self.critic_1_target = CriticNetwork(self.n_states, self.n_actions)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr, amsgrad=True)

        self.critic_2 = CriticNetwork(self.n_states, self.n_actions)
        self.critic_2_target = CriticNetwork(self.n_states, self.n_actions)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr, amsgrad=True)

        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic_1, self.critic_1_target)
        self._hard_update(self.critic_2, self.critic_2_target)

        self.buffer = ReplayBuffer(n_states, n_actions, int(buffer_size))
        self.batch_size = batch_size
        self.noise =  OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std_dev) * np.ones(1))
        self.tau = tau  # Target network update rate
        self.gamma = gamma  # Reward discount
        self.policy_noise = 0.2
        self.policy_noise_clip = 0.5
        self.iter = 0
        self.policy_freq = 2        
        self.last_actor_loss = 0

        if USE_CUDA: self._cuda()
    

    def remember(self, prev_state, action, state, reward, done):
        self.buffer.add(prev_state, action, state, reward, done)        
    
    def choose_action(self, state, random_act=False, noise=True):        
        if random_act:
            action = np.random.uniform(-1*np.ones(self.n_actions), np.ones(self.n_actions), self.n_actions)
        else:
            state = self._to_tensor(state, volatile=True, requires_grad=False).unsqueeze(0)                 
            action = self.actor(state)
            action = self._to_numpy(action).squeeze(0)            

        action += self.noise() if noise else 0        
        action = np.clip(action, -1., 1.)

        return action


    def learn(self):        
        self.iter += 1
       # Sample replay buffer 
        state, action, next_state, reward, not_done = self.buffer.sample(self.batch_size)


        with torch.no_grad():
            # Select action according to policy and clipped noise
            noise = (
                torch.rand_like(action) * self.policy_noise
            ).clamp(-self.policy_noise_clip, self.policy_noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-1, 1)
        
            # Compute target Q Value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss_function = nn.MSELoss()
        critic_1_loss = critic_loss_function(current_Q1, target_Q)
        critic_2_loss = critic_loss_function(current_Q2, target_Q)        

        # Optimize critic   
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()        
        critic_2_loss.backward()        
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.iter % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic_1(state, self.actor(state))            
            actor_loss = actor_loss.mean()
            

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()     

            # Update target networks
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic_1, self.critic_1_target, self.tau)
            self._soft_update(self.critic_2, self.critic_2_target, self.tau)
            self.last_actor_loss = actor_loss.item()            
            
        return self.last_actor_loss, critic_1_loss, critic_2_loss
        
       
        

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic_1.eval()
        self.critic_1_target.eval()
        self.critic_2.eval()
        self.critic_2_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic_1.train()
        self.critic_1_target.train()
        self.critic_2.train()
        self.critic_2_target.train()

    def load_weights(self, output):
        if output is None: return

        actor_checkpoint = torch.load('{}/td3_actor.chpt'.format(output))
        critic_1_checkpoint = torch.load('{}/td3_critic_1.chpt'.format(output))
        critic_2_checkpoint = torch.load('{}/td3_critic_2.chpt'.format(output))

        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])        
        self.actor_target.load_state_dict(actor_checkpoint['target_model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        self.critic_1.load_state_dict(critic_1_checkpoint['model_state_dict'])        
        self.critic_1_target.load_state_dict(critic_1_checkpoint['target_model_state_dict'])
        self.critic_1_optimizer.load_state_dict(critic_1_checkpoint['optimizer_state_dict'])
        self.critic_2.load_state_dict(critic_2_checkpoint['model_state_dict'])        
        self.critic_2_target.load_state_dict(critic_2_checkpoint['target_model_state_dict'])
        self.critic_2_optimizer.load_state_dict(critic_2_checkpoint['optimizer_state_dict'])
        
        return actor_checkpoint['steps'], actor_checkpoint['episodes']
    
    def save_weights(self, steps, episodes, output):
        torch.save({
                'steps': steps,
                'episodes': episodes,
                'model_state_dict': self.actor.state_dict(),
                'target_model_state_dict': self.actor_target.state_dict(),
                'optimizer_state_dict': self.actor_optimizer.state_dict(),
            },'{}/td3_actor.chpt'.format(output))

        torch.save({
                'steps': steps,
                'episodes': episodes,
                'model_state_dict': self.critic_1.state_dict(),
                'target_model_state_dict': self.critic_1_target.state_dict(),
                'optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            },'{}/td3_critic_1.chpt'.format(output))

        torch.save({
                'steps': steps,
                'episodes': episodes,
                'model_state_dict': self.critic_2.state_dict(),
                'target_model_state_dict': self.critic_2_target.state_dict(),
                'optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            },'{}/td3_critic_2.chpt'.format(output))

    def _cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic_1.cuda()
        self.critic_1_target.cuda()
        self.critic_2.cuda()
        self.critic_2_target.cuda()


    def _hard_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def _soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        pass


    def _to_tensor(self, ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
        if volatile:
            with torch.no_grad():
                return Variable(
                    torch.from_numpy(ndarray), requires_grad=requires_grad
                ).type(dtype)        
        else:
            return Variable(
                    torch.from_numpy(ndarray), requires_grad=requires_grad
                ).type(dtype)        
        

    def _to_numpy(self, var):
        return var.detach().cpu().data.numpy() if USE_CUDA else var.data.numpy()
    
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )