import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_ddpg.model import ActorNetwork, CriticNetwork
from pytorch_ddpg.ou import OUActionNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477



class DDPG(object):
    def __init__(
        self,
        n_states,
        n_actions,
        buffer_size=int(1e6),
        batch_size=512,
        noise_std_dev=0.2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=0.001,
        gamma=0.99
    ):
        self.n_states = n_states
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_states, n_actions).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, amsgrad=True)

        self.critic = CriticNetwork(n_states, n_actions).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, amsgrad=True)

        self.buffer = ReplayBuffer(n_states, n_actions, buffer_size)
        self.noise =  OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std_dev) * np.ones(1))
        self.tau = tau  # Target network update rate
        self.gamma = gamma  # Reward discount
        self.batch_size = batch_size

    def choose_action(self, state, random_act=False, noise=True):        
        self.actor.eval()
        if random_act:
            action = np.random.uniform(-1*np.ones(self.n_actions), np.ones(self.n_actions), self.n_actions)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state).cpu().data.numpy().flatten()
                     
        action += self.noise() if noise else 0        
        action = np.clip(action, -1., 1.)

        return action
  
    def remember(self, prev_state, action, reward, state, done):
        self.buffer.add(prev_state, action, reward, state, done)

    def learn(self):
        self.actor.train()
        # Sample replay buffer 
        state, action, next_state, reward, not_done = self.buffer.sample(self.batch_size, 0.8)
		

        with torch.no_grad():
            # Select action according to policy and add clipped noise          
            next_action = self.actor_target(next_state)

			# Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.l1_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        # Compute actor losse
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def load_weights(self, output):
        if output is None: return

        actor_checkpoint = torch.load('{}/actor.chpt'.format(output))
        critic_checkpoint = torch.load('{}/critic.chpt'.format(output))

        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])        
        self.actor_target.load_state_dict(actor_checkpoint['target_model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        self.critic.load_state_dict(critic_checkpoint['model_state_dict'])        
        self.critic_target.load_state_dict(critic_checkpoint['target_model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        
        return actor_checkpoint['steps'], actor_checkpoint['episodes']

    
    def save_weights(self, steps, episodes, output):
        torch.save({
                'steps': steps,
                'episodes': episodes,
                'model_state_dict': self.actor.state_dict(),
                'target_model_state_dict': self.actor_target.state_dict(),
                'optimizer_state_dict': self.actor_optimizer.state_dict(),
            },'{}/actor.chpt'.format(output))

        torch.save({
                'steps': steps,
                'episodes': episodes,
                'model_state_dict': self.critic.state_dict(),
                'target_model_state_dict': self.critic_target.state_dict(),
                'optimizer_state_dict': self.critic_optimizer.state_dict(),
            },'{}/critic.chpt'.format(output))

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), unbalance_gap=0.5):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        # temp variables
        self.unbalance_gap = unbalance_gap
        self.p_indices = [unbalance_gap/2]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, unbalance_p=False):


        # p_indices = None
        # if random.random() < unbalance_p:
        #     self.p_indices.extend((np.arange(self.size-len(self.p_indices))+1)
        #                           * self.unbalance_gap + self.p_indices[-1])
        #     p_indices = self.p_indices / np.sum(self.p_indices)
        
        # chosen_indices = np.random.choice(self.size,
        #                                   size=min(batch_size, self.size),
        #                                   replace=False,
        #                                   p=p_indices)

        # return (
        #     torch.FloatTensor(self.state[chosen_indices]).to(self.device),
        #     torch.FloatTensor(self.action[chosen_indices]).to(self.device),
        #     torch.FloatTensor(self.next_state[chosen_indices]).to(self.device),
        #     torch.FloatTensor(self.reward[chosen_indices]).to(self.device),
        #     torch.FloatTensor(self.not_done[chosen_indices]).to(self.device)
        # )
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )