import gym
from gym_uav_collision_avoidance.envs import MultiUAVWorld2D
import time

num_agent = 5
env = MultiUAVWorld2D(num_agents=num_agent)

observation, info = env.reset(return_info=True)

while True:
    n_action = []
    for i in range(num_agent):
        n_action.append(env.action_space.sample())

    observation, reward, done, info = env.step(n_action)
    time.sleep(0.02)
    env.render()
    

    print(observation[0])
    input("Press Enter to continue...")
    if done[0]:
        observation, info = env.reset(return_info=True)    
    
        

env.close()
