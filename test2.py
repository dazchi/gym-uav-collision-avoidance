from asyncore import read
from sys import maxsize
import time
import gym
import random
from matplotlib.style import available
import numpy as np
import tensorflow as tf
from gym_uav_collision_avoidance.envs import UAVWorld2D
from ddpg_tf2.model import Brain
from ddpg_tf2.common_definitions import UNBALANCE_P

env = UAVWorld2D()

n_actions = env.action_space.shape[-1]
total_episodes = 1000
evaluate = False

# Training parameters, set others in common_definition.py
CHECKPOINTS_PATH = "checkpoints/DDPG_"
TRAIN = False
USE_NOISE = True
SAVE_WEIGHTS = True
TOTAL_EPISODES = 1000
WARM_UP_EPISODES = 1
EPS_GREEDY = 0.95

brain = Brain(env.observation_space.shape[0], env.action_space.shape[0], 1, -1)

 # load weights if available
print("Loading weights from %s*, make sure the folder exists" % CHECKPOINTS_PATH)
brain.load_weights(CHECKPOINTS_PATH)


for ep in range(TOTAL_EPISODES):
    prev_state = env.reset()    
    done = False
    score = 0
    time_steps = 0    
    while not done:        
        if not TRAIN:
            no_random_act = True    
        else:
            no_random_act = (ep >= WARM_UP_EPISODES) and (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES)
        cur_act = brain.act(tf.expand_dims(prev_state, 0), _notrandom=no_random_act, noise=USE_NOISE)
        state, reward, done, _ = env.step(cur_act * env.action_space.high)
        brain.remember(prev_state, reward, state, int(done))

        # update weights
        if TRAIN:
            c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))

        score += reward
        prev_state = state
        
        env.render()
    if TRAIN:
        brain.save_weights(CHECKPOINTS_PATH)
    print('Ep = %d, Score = %.2f' % (ep, score))
        


