import sys
import os
import random
import time
import torch
import math
import gc
import numpy as np
from pytorch_ddpg.ddpg import DDPG
from torch.utils.tensorboard import SummaryWriter
from gym_uav_collision_avoidance.envs import UAVWorld2D
from torchviz import make_dot


torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

MODEL_PATH = './weights/ddpg'
WARM_UP_STEPS = 1000
MAX_EPISOED_STEPS = 3000
TOTAL_EPISODES = 1000
EVALUATE = False
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
ddpg = DDPG(n_observations, n_actions)


if EVALUATE or LOAD_MODEL:    
    ddpg.load_weights(MODEL_PATH)

if EVALUATE:
    ddpg.eval()
else:
    ddpg.train()

total_steps = 0
state, info = env.reset(return_info=True)

# o = torch.zeros(2,n_observations, dtype=torch.float, requires_grad=False, device=device)
# a = torch.zeros(2,n_actions, dtype=torch.float, requires_grad=False, device=device)
# x = ddpg.actor(o)
# make_dot(x, params=dict(list(ddpg.actor.named_parameters()))).render("actor_network", format="png")
# x = ddpg.critic(o, a)
# make_dot(x, params=dict(list(ddpg.critic.named_parameters()))).render("critic_network", format="png")

for eps in range(TOTAL_EPISODES): 
    score = 0
    eps_t = time.time()
    eps_steps = 0
    tr_list = []
    tca_list = []
    tl_list = []
    for steps in range(MAX_EPISOED_STEPS):
        random_action = (not LOAD_MODEL and (total_steps < WARM_UP_STEPS)) or (random.random() > EPSILON_GREEDY + (1-EPSILON_GREEDY)*eps/TOTAL_EPISODES) 


        action = ddpg.choose_action(state, random_action and not EVALUATE, noise=not EVALUATE)        
        v = (action[0]+1)/2 * np.linalg.norm(env.action_space.high)        
        theta = action[1] * math.pi
        converted_action = np.array([v*math.cos(theta), v*math.sin(theta)])
        
        new_state, reward, done, info = env.step(converted_action)                        
                
        ddpg.remember(state, action, reward, new_state, done)                

        if total_steps > WARM_UP_STEPS and not EVALUATE:                       
            actor_loss, critic_loss = ddpg.learn()
            tb_writer.add_scalar("Actor Loss/Steps", actor_loss, total_steps)
            tb_writer.add_scalar("Critic Loss/Steps", critic_loss, total_steps)
                                                
        state = new_state
        score += reward
        total_steps += 1   
        eps_steps += 1
        print("Steps = %d, Reward = %.3f, Score = %.3f" % (steps, reward, score), end='\r')        
        env.render()                
        if done:
            state, info = env.reset(return_info=True)
            break
    
    eps_t = time.time() - eps_t
    steps_per_sec = eps_steps / eps_t
    sys.stdout.write("\033[K")
    print("Total Steps = %d, Episode = %d, Score = %.3f, Steps Per Sec = %.2f" % (total_steps, eps, score, steps_per_sec))    
    tb_writer.add_scalar("Score/Episodes", score, eps)

    # print(torch.cuda.memory_summary())

    if not EVALUATE:
        ddpg.save_weights(MODEL_PATH)

    tb_writer.flush()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()    


env.close()
tb_writer.close()





    