import random
import time
import os
from matplotlib.style import available
import numpy as np
import tensorflow as tf
from gym_uav_collision_avoidance.envs import MultiUAVWorld2D
from ddpg_tf2.model import Brain
from ddpg_tf2.common_definitions import UNBALANCE_P
from ddpg_tf2.utils import Tensorboard

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

NUM_AGENT = 10
env = MultiUAVWorld2D(num_agents=NUM_AGENT)

# Training parameters, set others in common_definition.py
CHECKPOINTS_PATH = "checkpoints/multi/DDPG_"
TF_LOG_DIR = './logs/DDPG/'
TRAIN = True
USE_NOISE = True
SAVE_WEIGHTS = True
TOTAL_EPISODES = 3000
WARM_UP_EPISODES = 1
EPS_GREEDY = 0.95
D_SENSE = 30

brain = Brain(env.observation_space.shape[0], env.action_space.shape[0], 1, -1)
# load weights if available
print("Loading weights from %s*, make sure the folder exists" % CHECKPOINTS_PATH)
brain.load_weights(CHECKPOINTS_PATH)

# all the metrics
tensorboard = Tensorboard(log_dir=TF_LOG_DIR)
acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)


for ep in range(TOTAL_EPISODES):
    n_prev_state = env.reset()
    brain.noise.reset()
    acc_reward.reset_states()
    actions_squared.reset_states()
    Q_loss.reset_states()
    A_loss.reset_states()    
    done = False
    score = 0    
    ep_start_time = time.time()    
    while not done:        
        tt = time.time()
        if not TRAIN:
            no_random_act = True    
        else:
            no_random_act = (ep >= WARM_UP_EPISODES) and (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES)     
        
        n_cur_act = []
        n_cur_act_scaled = []
        for i in range(NUM_AGENT):
            t1 = time.time()
            cur_act = brain.act(tf.expand_dims(n_prev_state[i], 0), _notrandom=no_random_act, noise=USE_NOISE and TRAIN and (i==0))                 
            t1 = time.time()-t1
            n_cur_act.append(cur_act)
            n_cur_act_scaled.append(cur_act * env.action_space.high)
        n_state, n_reward, n_done, _ = env.step(n_cur_act_scaled)
    
        brain.remember(n_prev_state[0], n_cur_act[0], n_reward[0], n_state[0], int(n_done[0]))     
        
        # for i in range(NUM_AGENT -1):                               
        #     if n_done[i+1]: 
        #         continue
        #     brain.remember(n_prev_state[i+1], n_cur_act[i+1], n_reward[i+1], n_state[i+1], int(n_done[i+1]))        
                                   
        
        done = n_done[0]

        if env.steps > 1000:
            done = True

        # update weights
        if TRAIN:
            t2 = time.time()
            c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))            
            t2 = time.time() - t2
            Q_loss(c)
            A_loss(a)
        
        acc_reward(n_reward[0])
        actions_squared(np.square(n_cur_act[0]))
        score += n_reward[0]
        n_prev_state = n_state        
        tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

        
        env.render()
        print('t = %d, reward = %.2f, t1 = %.4f, t2 = %.4f, tt = %.4f' % (env.steps, n_reward[0], t1, t2, time.time()-tt), end='\r')
        
    
    if TRAIN:
        brain.save_weights(CHECKPOINTS_PATH)
    
    steps_per_second = env.steps / (time.time() - ep_start_time)
    print('Ep = %d, Score = %.2f, steps_per_second %.2f' % (ep, score ,steps_per_second))
        


