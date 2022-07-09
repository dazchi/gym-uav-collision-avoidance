from asyncore import read
from sys import maxsize
import time
import gym
from matplotlib.style import available
import numpy as np
import tensorflow as tf
from gym_uav_collision_avoidance.envs import UAVWorld2D
from ddpg import ddpg


env = UAVWorld2D()

n_actions = env.action_space.shape[-1]
total_episodes = 1000
evaluate = False

agent = ddpg.Agent(input_dims=(5,), n_actions=n_actions, batch_size=128, replay_buffer_size=10000, fc1=800, fc2=800, alpha=0.001, beta=0.002, tau=0.01)

if evaluate:
    observation = env.reset()
    state = np.array([
                    observation['normalized_agent_speed'][0],
                    observation['normalized_agent_speed'][1],
                    observation['normalized_target_relative_position'][0],
                    observation['normalized_target_relative_position'][1],
                    # observation['normalized_relative_target_theta'],
                    # observation['normalized_agent_theta'],
                    observation['normalized_delta_theta'],
                ])    
    agent.choose_action(state, evaluate)            
    agent.load_models(evaluate)

for i in range(total_episodes):
    observation = env.reset()
    done = False
    score = 0
    time_steps = 0
    print("episode = " + str(i))
    while not done:
        state = np.array([
                    observation['normalized_agent_speed'][0],
                    observation['normalized_agent_speed'][1],
                    observation['normalized_target_relative_position'][0],
                    observation['normalized_target_relative_position'][1],
                    # observation['normalized_relative_target_theta'],
                    # observation['normalized_agent_theta'],
                    observation['normalized_delta_theta'],
                ])    
        action = agent.choose_action(state, evaluate)        
        observation, reward, done, info = env.step( action * env.action_space.high)
        new_state = np.array([
                    observation['normalized_agent_speed'][0],
                    observation['normalized_agent_speed'][1],
                    observation['normalized_target_relative_position'][0],
                    observation['normalized_target_relative_position'][1],
                    # observation['normalized_relative_target_theta'],
                    # observation['normalized_agent_theta'],
                    observation['normalized_delta_theta'],
                ])
        # print(new_state)
        agent.remember(state, action, reward, new_state, done)
        if not evaluate:                                
            agent.learn()  
        score += reward
        time_steps += 1
        # print(time_steps)
        if time_steps > 1000:
            done = True
            print('Ending episode becauce timeout')
        env.render()
    agent.save_models() 
    print('Ep = %d, Score = %.2f' % (i, score))
        


