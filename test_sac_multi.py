import sys
import os
import random
import time
from turtle import done
import torch
import math
import gc
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gym_uav_collision_avoidance.envs import MultiUAVWorld2D
from pytorch_sac_temp.sac import SAC
from pytorch_sac_temp.replay_memory import ReplayMemory
from torchviz import make_dot


MODEL_PATH = './weights/sac_multi'
WARM_UP_STEPS = 3000
MAX_EPISOED_STEPS = 3000
TOTAL_EPISODES = 20000
BATCH_SIZE = 256
EVALUATE = False
UPDATE_PER_STEP = 1
LOAD_MODEL = False
NUM_AGENTS = 12
EPSILON_GREEDY = 0.95


# If GPU is to be used
torch.set_flush_denormal(True)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device = %s' % device)


env = MultiUAVWorld2D(num_agents=NUM_AGENTS)
tb_writer = SummaryWriter()

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

agents = []
for i in range(NUM_AGENTS):
    agents.append(SAC(n_observations, n_actions))

if EVALUATE or LOAD_MODEL:    
    for i in range(NUM_AGENTS):
        agents[i].load_checkpoint(MODEL_PATH, EVALUATE)
    
# o = torch.zeros(2,n_observations, dtype=torch.float, requires_grad=False, device=device)
# a = torch.zeros(2,n_actions, dtype=torch.float, requires_grad=False, device=device)
# x = ddpg.actor(o)
# make_dot(x, params=dict(list(ddpg.actor.named_parameters()))).render("actor_network", format="png")
# x = ddpg.critic(o, a)
# make_dot(x, params=dict(list(ddpg.critic.named_parameters()))).render("critic_network", format="png")

# Memory
memory = ReplayMemory(int(1e6))
updates = 0
total_steps = 0
states, _ = env.reset(return_info=True)
for eps in range(TOTAL_EPISODES): 
    score = 0
    eps_t = time.time()
    eps_steps = 0
    for steps in range(MAX_EPISOED_STEPS):
        actions = []
        converted_actions = []

        for i in range(NUM_AGENTS):                
            if total_steps < WARM_UP_STEPS and not EVALUATE and not LOAD_MODEL: 
                action = np.random.uniform(low=-1, high=1, size=(n_actions,))                       
            else:
                action = agents[i].select_action(states[i], evaluate=EVALUATE)        

            v = (action[0]/2+0.5) * np.linalg.norm(env.action_space.high)     
            theta = action[1] * math.pi
        
            converted_action = np.array([v*math.cos(theta), v*math.sin(theta)])
            
            actions.append(action)
            converted_actions.append(converted_action)
        
        if len(memory) > BATCH_SIZE and not EVALUATE:
             # Number of updates per step in environment
            for i in range(UPDATE_PER_STEP):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agents[0].update_parameters(memory, BATCH_SIZE, updates)
                for i in range(1, NUM_AGENTS):            
                    agents[i].policy.load_state_dict(agents[0].policy.state_dict())
                tb_writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                tb_writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                tb_writer.add_scalar('loss/policy', policy_loss, updates)
                tb_writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                tb_writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_states, rewards, dones, _ = env.step(converted_actions) # Step
    
        memory.push(states[0], actions[0], rewards[0], next_states[0], float(not dones[0])) # Append transition to memory                                                                                 
        for i in range(1, NUM_AGENTS):            
            memory.push(states[i], actions[i], rewards[i], next_states[i], float(not dones[i])) # Append transition to memory                                                                                 
                                                      
        states = next_states
        score += rewards[0]
        total_steps += 1   
        eps_steps += 1
        print("Steps = %d, Reward = %.3f, Score = %.3f" % (steps, rewards[0], score), end='\r')        
        env.render()       
        if not EVALUATE:            
            if dones[0]:            
                break
        else:
            # print(dones)
            if all(dones):
                break            
        
    success_rate = env.target_reach_count / NUM_AGENTS
    collision_rate = env.collision_count / NUM_AGENTS
    
    state, info = env.reset(return_info=True)
    eps_t = time.time() - eps_t
    steps_per_sec = eps_steps / eps_t
    sys.stdout.write("\033[K")
    print("Total Steps = %d, Episode = %d, Score = %.3f, Steps Per Sec = %.2f, SR = %.2f, CR = %.2f" % (total_steps, eps, score, steps_per_sec, success_rate, collision_rate))    
    tb_writer.add_scalar("Score/Episodes", score, eps)
    tb_writer.add_scalar("SR/Episodes", success_rate, eps)
    tb_writer.add_scalar("CR/Episodes", collision_rate, eps)

    # print(torch.cuda.memory_summary())

    if not EVALUATE:
        agents[0].save_checkpoint(MODEL_PATH)
   

    tb_writer.flush()

    

env.close()
tb_writer.close()





    