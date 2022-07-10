import gym
import pygame
import numpy as np

class UAVAgent():
    def __init__(self, color, max_speed, max_acceleraion, tau):
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
    
    
    def step(self, action):
        dv = np.clip((action - self.velocity_prev)/self.tau, -self.max_acceleration, self.max_acceleration)
        self.velocity = np.clip(self.velocity_prev + dv * self.tau, -self.max_speed, self.max_speed)
        dx = self.velocity * self.tau        
        self.location += dx
        self.velocity_prev = self.velocity