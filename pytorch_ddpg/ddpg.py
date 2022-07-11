import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from pytorch_ddpg.model import ActorNetwork, CriticNetwork
from pytorch_ddpg.buffer import ReplayBuffer
from pytorch_ddpg.ou import OUActionNoise

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
UNBALANCE_P = 0.8

class DDPG(object):
    def __init__(self, n_states, n_actions, buffer_size=1e6, batch_size=256, noise_std_dev=0.2, actor_lr=1e-4, critic_lr=1e-3, tau=0.001, gamma=0.99):
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

        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.noise =  OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std_dev) * np.ones(1))
        self.tau = tau  # Target network update rate
        self.gamma = gamma  # Reward discount
       
    # 
        if USE_CUDA: self._cuda()
    

    def remember(self, prev_state, action, reward, state, done):
        self.buffer.append(prev_state, action, reward, state, done)        
    
    def choose_action(self, state, random_act=False, noise=True):
        if random_act:
            action = np.random.uniform(-1*np.ones(self.n_actions), np.ones(self.n_actions), self.n_actions)
        else:
            state = self._to_tensor(state).unsqueeze(0)                
            action = self.actor(state)
            action = self._to_numpy(action).squeeze(0)

        action = action + self.noise() if noise else 0
        
        action = np.clip(action, -1., 1.)
    
        return action


    def learn(self):
        batch = self.buffer.get_batch(unbalance_p=UNBALANCE_P)
                
        state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch = zip(*batch)
  
        state_batch = self._to_tensor(np.asarray(state_batch))
        action_batch = self._to_tensor(np.asarray(action_batch))        
        reward_batch = self._to_tensor(np.asarray(reward_batch))
        next_state_batch = self._to_tensor(np.asarray(next_state_batch), volatile=True)
        done_batch = self._to_tensor(np.asarray(done_batch))

        
        # Update critic network
        y = reward_batch + self.gamma * (1 - done_batch) * self.critic_target(next_state_batch, self.actor_target(next_state_batch))                       
        q = self.critic(state_batch, action_batch)
        
        self.critic.zero_grad()
        # loss_function = nn.MSELoss()
        loss_function = nn.L1Loss()     # Mean Absolute Loss                
        critic_loss = loss_function(y, q)
        critic_loss.backward()
        self.critic_optimizer.step()        

        # Update actor network
        self.actor.zero_grad()
        actor_loss = -self.critic(state_batch, self.actor(state_batch))
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
                
        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)        

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )
        self.actor_target.load_state_dict(
            torch.load('{}/actor_target.pkl'.format(output))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )
        self.critic_target.load_state_dict(
            torch.load('{}/critic_target.pkl'.format(output))
        )

    def save_weights(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.actor_target.state_dict(),
            '{}/actor_target.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        torch.save(
            self.critic_target.state_dict(),
            '{}/critic_target.pkl'.format(output)
        )

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
        return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()
    
   