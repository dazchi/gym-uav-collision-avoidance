from asyncore import read
from sys import maxsize
import random
from matplotlib.style import available
import numpy as np
import tensorflow as tf
from gym_uav_collision_avoidance.envs import UAVWorld2D
from ddpg_tf2.model import Brain
from ddpg_tf2.common_definitions import UNBALANCE_P
from ddpg_tf2.utils import Tensorboard

env = UAVWorld2D()

# Training parameters, set others in common_definition.py
CHECKPOINTS_PATH = "checkpoints/DDPG_"
TF_LOG_DIR = './logs/DDPG/'
TRAIN = False
USE_NOISE = True
SAVE_WEIGHTS = True
TOTAL_EPISODES = 1000
WARM_UP_EPISODES = 0
EPS_GREEDY = 0.95

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
    prev_state = env.reset()
    brain.noise.reset()
    acc_reward.reset_states()
    actions_squared.reset_states()
    Q_loss.reset_states()
    A_loss.reset_states()    
    done = False
    score = 0
    time_steps = 0    
    while not done:        
        if not TRAIN:
            no_random_act = True    
        else:
            no_random_act = (ep >= WARM_UP_EPISODES) and (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES)
        cur_act = brain.act(tf.expand_dims(prev_state, 0), _notrandom=no_random_act, noise=USE_NOISE and TRAIN)
        state, reward, done, _ = env.step(cur_act * env.action_space.high)
        brain.remember(prev_state, reward, state, int(done))

        # update weights
        if TRAIN:
            c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
            Q_loss(c)
            A_loss(a)
        acc_reward(reward)
        actions_squared(np.square(cur_act))
        score += reward
        prev_state = state        
        env.render()
        tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)
    if TRAIN:
        brain.save_weights(CHECKPOINTS_PATH)
    print('Ep = %d, Score = %.2f' % (ep, score))
        


