import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		
		self.actor = Actor(state_dim, action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = 1
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq		


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def update_parameters(self, memory, batch_size, updates):		

		# Sample replay buffer 
		# Sample a batch from memory
		state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

		state_batch = torch.FloatTensor(state_batch).to(device)
		next_state_batch = torch.FloatTensor(next_state_batch).to(device)
		action_batch = torch.FloatTensor(action_batch).to(device)
		reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
		mask_batch = torch.FloatTensor(mask_batch).to(device).unsqueeze(1)

		state, action, next_state, reward, not_done = state_batch, action_batch, next_state_batch, reward_batch, mask_batch

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if updates % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save_checkpoint(self, ckpt_path):
		if not os.path.exists(ckpt_path):
			os.makedirs(ckpt_path)

		torch.save({
					'actor_state_dict': self.actor.state_dict(),
					'actor_target_state_dict': self.actor_target.state_dict(),
					'critic_state_dict': self.critic.state_dict(),
					'critic_target_state_dict': self.critic_target.state_dict(),
					'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
					'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
						}, '{}/weights.chpt'.format(ckpt_path))		


	def load_checkpoint(self, ckpt_path, evaluate):
		print('Loading models from {}'.format(ckpt_path))

		checkpoint = torch.load('{}/weights.chpt'.format(ckpt_path))
		self.actor.load_state_dict(checkpoint['actor_state_dict'])
		self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
		self.critic.load_state_dict(checkpoint['critic_state_dict'])
		self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
		self.self.load_state_dict(checkpoint['actor_optimizer_state_dict'])
		self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])		
		
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

