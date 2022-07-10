import gym
from gym_uav_collision_avoidance.envs import MultiUAVWorld2D
import time

num_agent = 4
env = MultiUAVWorld2D(num_agents=num_agent)

observation, info = env.reset(return_info=True)

while True:
    n_action = []
    for i in range(num_agent):
        n_action.append(env.action_space.sample())

    observation, reward, done, info = env.step(n_action)
    env.render()
    
    # print(observation)
    if done:
        observation, info = env.reset(return_info=True)    
    
        

env.close()
