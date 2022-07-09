import gym
from gym_uav_collision_avoidance.envs import UAVWorld2D
import time


env = UAVWorld2D()

observation, info = env.reset(return_info=True)

while True:
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        observation, info = env.reset(return_info=True)    
        

env.close()
