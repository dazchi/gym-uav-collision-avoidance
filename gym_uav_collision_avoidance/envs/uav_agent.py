from math import fabs
import gym
import pygame
import numpy as np

class UAVAgent():
    def __init__(self, color, max_speed=10, max_acceleraion=4, tau=0.02):
        self.color = color
        self.max_speed = np.array([max_speed, max_speed])
        self.max_acceleration = np.array([max_acceleraion, max_acceleraion])    
        self.tau = tau        
        self.location = np.zeros(2)
        self.velocity = np.zeros(2)
        self.velocity_prev = np.zeros(2)        
        self.target_location = np.zeros(2)
        self.init_distance = 0
        self.prev_distance = 0
        self.done = False
        self.collided = False        
    
    
    def step(self, action):
        if self.done:
            return 0, 0
        dv = np.clip((action - self.velocity_prev)/self.tau, -self.max_acceleration, self.max_acceleration)
        self.velocity = np.clip(self.velocity_prev + dv * self.tau, -self.max_speed, self.max_speed)
        dx = self.velocity * self.tau        
        self.location += dx
        self.velocity_prev = self.velocity

        prev_distance = self.prev_distance
        distance = np.linalg.norm(self.target_location - self.location)
        self.prev_distance = distance

        return prev_distance, distance

    def finish(self):
        self.done = True
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * 0.001

    def uavs_in_range(self, uav_agents, d_sense=30):
        uavs = []
        relative_distances = []
        for i in range(len(uav_agents)):
            target_agent = uav_agents[i]
            if target_agent == self:
                continue
            distance = np.linalg.norm(target_agent.location - self.location)
            if np.linalg.norm(target_agent.location - self.location) < d_sense:
                uavs.append(target_agent)
                relative_distances.append(distance)
        
        # Sort UAVs with relative distances 
        if len(uavs) == 0:
            return []
        else:
            uavs = np.array(uavs)
            relative_distances = np.array(relative_distances)
            inds = relative_distances.argsort()
            sorted_uavs = uavs[inds]
        return sorted_uavs.tolist()
        
       