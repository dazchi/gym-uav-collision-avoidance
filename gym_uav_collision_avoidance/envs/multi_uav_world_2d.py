from cmath import cos, pi
import math
import colorsys
from random import randrange
from turtle import position
from cv2 import normalize, resizeWindow
import gym
from gym import spaces
import pygame
import numpy as np
from gym_uav_collision_avoidance.envs.uav_agent import UAVAgent


class MultiUAVWorld2D(gym.Env):
    metadata = {"render_fps": 1000}

    def __init__(self, x_size=100.0, y_size=100.0, num_agents=4, max_speed=12.0, max_acceleration=5.0):
        self.x_size = x_size # size of x dimension
        self.y_size = y_size # size of y dimension
        self.num_agents = num_agents
        self.map_diagonal_size = np.linalg.norm([x_size, y_size])
        self.map_dimension = np.array([x_size, y_size])
        self.min_location = np.array([-x_size/2.0, -y_size/2.0])
        self.max_location = np.array([x_size/2.0, y_size/2.0])        
        self.max_speed = np.array([max_speed, max_speed]) # Maximum speed of uav
        self.min_speed = np.array([-max_speed, -max_speed]) # Minimum speed of uav
        self.max_acceleratoin = np.array([max_acceleration, max_acceleration]) # Maximum speed of uav
        self.min_acceleratoin = np.array([-max_acceleration, -max_acceleration]) # Minimum speed of uav
        self.max_window_size = 800  # The size of the PyGame window
        self.tau = 0.02 # seconds between state updates
        self.collider_radius = 0.5 # Size of the UAV collider
        if x_size > y_size:
            self.window_size_x = self.max_window_size
            self.window_size_y = self.max_window_size / x_size * y_size            
        else:
            self.window_size_y = self.max_window_size
            self.window_size_x = self.max_window_size / y_size * x_size        
        
        self.agent_list = []        
        for i in range(num_agents):
            hue = i / num_agents
            (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = int(255 * r), int(255 * g), int(255 * b)
            self.agent_list.append(UAVAgent(color=color, max_speed=max_speed, max_acceleraion=max_acceleration, tau=self.tau))            
                
               
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)       
        self.action_space = spaces.Box(-max_speed, max_speed, shape=(2,), dtype=np.float32)

        
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self, agent):
        normalized_agent_speed = agent.velocity / agent.max_speed
        normalized_target_relative_position = (agent.target_location - agent.location) / self.map_diagonal_size
        target_theta = math.atan2((agent.target_location - agent.location)[1],(agent.target_location - agent.location)[0])
        agent_theta = math.atan2(agent.velocity[1], agent.velocity[0])    
        delta_theta = target_theta - agent_theta
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta)) 
        normalized_delta_theta = delta_theta / math.pi

        return np.array([normalized_agent_speed[0],
                         normalized_agent_speed[1],
                         normalized_target_relative_position[0],
                         normalized_target_relative_position[1],
                        #  normalized_delta_theta
                        ])        

    def _get_info(self):
        return {
            "distance": 0
        }

    def reset(self, return_info=False, options=None):
        # Choose the UoI's location uniformly at random
        for i in range(self.num_agents):
            self.agent_list[i].location = np.random.uniform(self.min_location, high=self.max_location, size=(2,)).astype(np.float32)
            self.agent_list[i].velocity = np.random.uniform(self.min_speed, high=self.max_speed, size=(2,)).astype(np.float32)        
            self.agent_list[i].velocity_prev = self.agent_list[i].velocity                    

        # Choose the goal's location uniformly at random
        for i in range(self.num_agents):        
            self.agent_list[i].target_location = np.random.uniform(self.min_location, high=self.max_location, size=(2,)).astype(np.float32)                
            self.agent_list[i].init_distance = np.linalg.norm(self.agent_list[i].target_location - self.agent_list[i].location)
            self.agent_list[i].prev_distance = self.agent_list[i].init_distance

        
        self.steps = 0

        n_observation = []
        for i in range(self.num_agents):
            n_observation.append(self._get_obs(self.agent_list[i]))
        
        info = self._get_info()
        return (n_observation, info) if return_info else n_observation

    def step(self, n_action):           
        for i in range(self.num_agents):
            self.agent_list[i].step(n_action[i])
        
        n_reward = []
        n_done = []
        for i in range(self.num_agents):
            clipped_location = np.clip(self.agent_list[i].location, self.min_location , self.max_location)
            distance = np.linalg.norm(self.agent_list[i].target_location - self.agent_list[i].location)      

            reward = 0 
            reward -= 1 / self.agent_list[i].init_distance        
            reward += 10 * (self.agent_list[i].prev_distance - distance)
            delta_theta = math.atan2((self.agent_list[i].target_location - self.agent_list[i].location)[1],
                (self.agent_list[i].target_location - self.agent_list[i].location)[0]) - math.atan2(self.agent_list[i].velocity[1],self.agent_list[i].velocity[0])
            delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))            
            reward -= 0.1 * abs(delta_theta)

            if distance < 0.5:  # An episode is done if the agent has reached the target        
                done = True            
                reward += 1000
            elif (clipped_location != self.agent_list[i].location).any():  # An episode is done if the agent has gone out of box            
                done = True            
                reward -= 1500                        
            else:
                done = False
            self.agent_list[i].prev_distance = distance  
            n_reward.append(reward)
            n_done.append(done)
        
        n_observation = []
        for i in range(self.num_agents):
            n_observation.append(self._get_obs(self.agent_list[i]))
        
        info = self._get_info()                  
        self.steps += 1      
        
        self._prev_distance = distance                
        return n_observation, n_reward, n_done, info

    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        canvas.fill((255, 255, 255))
        pixel_per_meter = self.window_size_x / self.x_size
        object_render_size = 10
        
        for i in range(self.num_agents):
            # First we draw the target
            target_render_location = (self.agent_list[i].target_location + np.array([self.x_size/2, self.y_size/2])) * pixel_per_meter 
            target_render_location[1] = self.window_size_y - target_render_location[1] 
            pygame.draw.rect(
                canvas,
                self.agent_list[i].color,
                pygame.Rect(
                    target_render_location,
                    (object_render_size, object_render_size),
                ),
            )
            
            
            # Now we draw the agent
            agent_render_location = (self.agent_list[i].location + np.array([self.x_size/2, self.y_size/2])) * pixel_per_meter 
            agent_render_location[1] = self.window_size_y - agent_render_location[1] 
            pygame.draw.circle(
                canvas,
                self.agent_list[i].color,
                agent_render_location,
                object_render_size,
            )
            theta_v = math.atan2(-self.agent_list[i].velocity[1], self.agent_list[i].velocity[0])
            pygame.draw.line(canvas, (0, 0, 0), agent_render_location, 
                agent_render_location+(object_render_size*math.cos(theta_v), object_render_size*math.sin(theta_v)),
                width=3
            )
        
        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
