import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.cuda import amp
from pytorch_ddpg_temp.model import ActorNetwork, CriticNetwork
from pytorch_ddpg_temp.ou import OUActionNoise

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
UNBALANCE_P = 0.8

class DDPG(object):
    def __init__(self, n_states, n_actions, noise_std_dev=0.2, actor_lr=1e-4, critic_lr=1e-3, tau=5e-3, gamma=0.99):        
        self.n_states = n_states
        self.n_actions = n_actions

        self.actor = ActorNetwork(self.n_states, self.n_actions)
        self.actor_target = ActorNetwork(self.n_states, self.n_actions)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, amsgrad=True)

        self.critic = CriticNetwork(self.n_states, self.n_actions)
        self.critic_target = CriticNetwork(self.n_states, self.n_actions)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, amsgrad=True)

        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic, self.critic_target)
        
        self.noise =  OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std_dev) * np.ones(1))
        self.tau = tau  # Target network update rate
        self.gamma = gamma  # Reward discount
       
    # 
        if USE_CUDA: self._cuda()


    def select_action(self, state, evaluate = False):        
        noise = not evaluate
        state = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)     
        action = self.actor(state)
        action = action.detach().cpu().numpy()[0]                      
        action += self.noise() if noise else 0        
        action = np.clip(action, -1., 1.)

        return action


    def update_parameters(self, memory, batch_size):
        # self.actor.train()
        
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(DEVICE)
        next_state_batch = torch.FloatTensor(next_state_batch).to(DEVICE)
        action_batch = torch.FloatTensor(action_batch).to(DEVICE)
        reward_batch = torch.FloatTensor(reward_batch).to(DEVICE).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(DEVICE).unsqueeze(1)           
                          
        # Update critic network
        y = reward_batch + self.gamma * mask_batch * self.critic_target(next_state_batch, self.actor_target(next_state_batch))                   
        q = self.critic(state_batch, action_batch)        
        # self.critic.zero_grad(set_to_none=True)
        for param in self.critic.parameters():
            param.grad = None
        loss_function = nn.MSELoss()
        loss_function = nn.L1Loss()     # Mean Absolute Loss                
        critic_loss = loss_function(y, q)
        critic_loss.backward()
        self.critic_optimizer.step()        

        # Update actor network
        # self.actor.zero_grad(set_to_none=True)
        for param in self.actor.parameters():
            param.grad = None
        actor_loss = -self.critic(state_batch, self.actor(state_batch))
        actor_loss = actor_loss.mean()
        actor_loss.backward()        
        self.actor_optimizer.step()
                        
        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)                 
              
        return actor_loss.item(), critic_loss.item()
        

    # def eval(self):
    #     self.actor.eval()
    #     self.actor_target.eval()
    #     self.critic.eval()
    #     self.critic_target.eval()

    # def train(self):
    #     self.actor.train()
    #     self.actor_target.train()
    #     self.critic.train()
    #     self.critic_target.train()
  
    def load_checkpoint(self, output, evaluate=True):
        if output is None: return

        actor_checkpoint = torch.load('{}/actor.chpt'.format(output))
        critic_checkpoint = torch.load('{}/critic.chpt'.format(output))

        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])        
        self.actor_target.load_state_dict(actor_checkpoint['target_model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        self.critic.load_state_dict(critic_checkpoint['model_state_dict'])        
        self.critic_target.load_state_dict(critic_checkpoint['target_model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        
        if evaluate:
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic_target.train()

    def save_checkpoint(self, output):
        torch.save({        
                'model_state_dict': self.actor.state_dict(),
                'target_model_state_dict': self.actor_target.state_dict(),
                'optimizer_state_dict': self.actor_optimizer.state_dict(),
            },'{}/actor.chpt'.format(output))

        torch.save({            
                'model_state_dict': self.critic.state_dict(),
                'target_model_state_dict': self.critic_target.state_dict(),
                'optimizer_state_dict': self.critic_optimizer.state_dict(),
            },'{}/critic.chpt'.format(output))

    def _cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()


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
    
   