import gym
from gym_uav_collision_avoidance.envs import UAVWorld2D

env = UAVWorld2D()
env.action_space.seed(100)

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        observation, info = env.reset(return_info=True)

env.close()
