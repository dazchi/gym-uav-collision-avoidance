import random
from matplotlib.style import available
import numpy as np
import tensorflow as tf
from gym_uav_collision_avoidance.envs import MultiUAVWorld2D
from ddpg_tf2.model import Brain
from ddpg_tf2.common_definitions import UNBALANCE_P
from ddpg_tf2.utils import Tensorboard

NUM_AGENT = 10
env = MultiUAVWorld2D(num_agents=NUM_AGENT)

# Training parameters, set others in common_definition.py
CHECKPOINTS_PATH = "checkpoints/multi/DDPG_"
TF_LOG_DIR = './logs/DDPG/'
TRAIN = True
USE_NOISE = True
SAVE_WEIGHTS = True
TOTAL_EPISODES = 3000
WARM_UP_EPISODES = 3
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
    while not done:        
        if not TRAIN:
            no_random_act = True    
        else:
            no_random_act = (ep >= WARM_UP_EPISODES) and (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES)     
        
        n_cur_act = []
        n_cur_act_scaled = []
        for i in range(NUM_AGENT):
            cur_act = brain.act(tf.expand_dims(n_prev_state[i], 0), _notrandom=no_random_act, noise=USE_NOISE and TRAIN and (i==0))                                                    
            n_cur_act.append(cur_act)
            n_cur_act_scaled.append(cur_act * env.action_space.high)
        n_state, n_reward, n_done, _ = env.step(n_cur_act_scaled)
    
        # for i in range(NUM_AGENT):                               
        #     brain.remember(n_prev_state[i], n_reward[i], n_state[i], int(n_done[i]))        
        brain.remember(n_prev_state[0], n_cur_act[0], n_reward[0], n_state[0], int(n_done[0]))        
        
        print('reward = %f' % n_reward[0], end='\r')
        done = n_done[0]

        if env.steps > 1000:
            done = True

        # update weights
        if TRAIN:
            c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
            Q_loss(c)
            A_loss(a)
        acc_reward(n_reward[0])
        actions_squared(np.square(n_cur_act[0]))
        score += n_reward[0]
        n_prev_state = n_state                
        env.render()
        tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)
    if TRAIN:
        brain.save_weights(CHECKPOINTS_PATH)
    print('Ep = %d, Score = %.2f' % (ep, score))
        


