import sys
import os
import random
import time
import torch
import math
import gc
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gym_uav_collision_avoidance.envs import UAVWorld2D
from pytorch_sac_temp.sac import SAC
from pytorch_sac_temp.replay_memory import ReplayMemory
from torchviz import make_dot


MODEL_PATH = './weights/sac'
WARM_UP_STEPS = 10000
MAX_EPISOED_STEPS = 2000
TOTAL_EPISODES = 1000
BATCH_SIZE = 256
EVALUATE = False
UPDATE_PER_STEP = 1
LOAD_MODEL = False

EPSILON_GREEDY = 0.95


# If GPU is to be used
torch.set_flush_denormal(True)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device = %s' % device)


env = UAVWorld2D()
tb_writer = SummaryWriter()

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
agent = SAC(n_observations, n_actions)


if EVALUATE or LOAD_MODEL:    
    agent.load_checkpoint(MODEL_PATH, EVALUATE)
    


total_steps = 0
state, info = env.reset(return_info=True)

# o = torch.zeros(2,n_observations, dtype=torch.float, requires_grad=False, device=device)
# a = torch.zeros(2,n_actions, dtype=torch.float, requires_grad=False, device=device)
# x = ddpg.actor(o)
# make_dot(x, params=dict(list(ddpg.actor.named_parameters()))).render("actor_network", format="png")
# x = ddpg.critic(o, a)
# make_dot(x, params=dict(list(ddpg.critic.named_parameters()))).render("critic_network", format="png")

# Memory
memory = ReplayMemory(int(1e6))
updates = 0

for eps in range(TOTAL_EPISODES): 
    score = 0
    eps_t = time.time()
    eps_steps = 0
    tr_list = []
    tca_list = []
    tl_list = []
    for steps in range(MAX_EPISOED_STEPS):
        random_action = (not LOAD_MODEL and (total_steps < WARM_UP_STEPS)) or (random.random() > EPSILON_GREEDY + (1-EPSILON_GREEDY)*eps/TOTAL_EPISODES) 

        if total_steps < WARM_UP_STEPS and not EVALUATE: 
            action = np.random.uniform(low=-1, high=1, size=(n_actions,))         
        else:
            action = agent.select_action(state, evaluate=EVALUATE)
        
        v = (action[0]/2+0.5) * env.action_space.high[0]
        theta = action[1] * math.pi
        
        converted_action = np.array([v*math.cos(theta), v*math.sin(theta)])
        
        if len(memory) > BATCH_SIZE and not EVALUATE:
             # Number of updates per step in environment
            for i in range(UPDATE_PER_STEP):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, BATCH_SIZE, updates)

                tb_writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                tb_writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                tb_writer.add_scalar('loss/policy', policy_loss, updates)
                tb_writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                tb_writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(converted_action) # Step
        mask = float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory                                                                                 
                                                      
        state = next_state
        score += reward
        total_steps += 1   
        eps_steps += 1
        print("Steps = %d, Reward = %.3f, Score = %.3f" % (steps, reward, score), end='\r')        
        env.render()                
        if done:            
            break
    
    state, info = env.reset(return_info=True)
    eps_t = time.time() - eps_t
    steps_per_sec = eps_steps / eps_t
    sys.stdout.write("\033[K")
    print("Total Steps = %d, Episode = %d, Score = %.3f, Steps Per Sec = %.2f" % (total_steps, eps, score, steps_per_sec))    
    tb_writer.add_scalar("Score/Episodes", score, eps)

    # print(torch.cuda.memory_summary())

    if not EVALUATE:
        agent.save_checkpoint(MODEL_PATH)        

    tb_writer.flush()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()    


env.close()
tb_writer.close()





    