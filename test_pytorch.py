import sys
import random
import torch
import numpy as np
from pytorch_ddpg.ddpg import DDPG
from gym_uav_collision_avoidance.envs import UAVWorld2D
from torchviz import make_dot

MODEL_PATH = './weights/ddpg'
WARM_UP_STEPS = 1000
MAX_EPISOED_STEPS = 300000
TOTAL_EPISODES = 1000
EVALUATE = False
LOAD_MODEL = False

EPSILON_GREEDY = 0.95

# If GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device = %s' % device)

env = UAVWorld2D()

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
ddpg = DDPG(n_observations, n_actions)


if EVALUATE or EVALUATE:    
    ddpg.load_weights(MODEL_PATH)

if EVALUATE:
    ddpg.eval()    

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
    for steps in range(MAX_EPISOED_STEPS):
        random_action = (not LOAD_MODEL and (total_steps < WARM_UP_STEPS)) or (random.random() > EPSILON_GREEDY + (1-EPSILON_GREEDY)*eps/TOTAL_EPISODES)                
        action = ddpg.choose_action(state, random_action, noise=not EVALUATE)        
        new_state, reward, done, info = env.step(action * env.action_space.high)                        
        
        ddpg.remember(state, action, reward, new_state, done)        

        if total_steps > WARM_UP_STEPS and not EVALUATE:                    
            ddpg.learn()
                                                
        state = new_state
        score += reward
        total_steps += 1    
        print("Steps = %d, Reward = %.3f, Score = %.3f" % (steps, reward, score), end='\r')
        env.render()        

        if done:
            state, info = env.reset(return_info=True)
            break
    
    sys.stdout.write("\033[K")
    print("Total Steps = %d, Episode = %d, Score = %.3f" % (total_steps, eps, score))

    if not EVALUATE:
        ddpg.save_weights(MODEL_PATH)

env.close()




