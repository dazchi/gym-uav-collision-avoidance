from cmath import cos, pi
import math
from turtle import position
from cv2 import normalize, resizeWindow
import gym
from gym import spaces
import pygame
import numpy as np


class UAVWorld2D(gym.Env):
    metadata = {"render_fps": 1000}

    def __init__(self, x_size=100.0, y_size=100.0, agent_num=4, max_speed=12.0, max_acceleration=5.0):
        self.x_size = x_size # size of x dimension
        self.y_size = y_size # size of y dimension
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
        if x_size > y_size:
            self.window_size_x = self.max_window_size
            self.window_size_y = self.max_window_size / x_size * y_size            
        else:
            self.window_size_y = self.max_window_size
            self.window_size_x = self.max_window_size / y_size * x_size        
        

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         # "agent": spaces.Box(low=np.concatenate((self.min_location, self.min_speed)),
        #         #     high=np.concatenate((self.max_location, self.max_speed)), dtype=np.float32),
        #         # "target": spaces.Box(low=self.min_location, high=self.max_location, dtype=np.float32),
        #         # "agent_speed": spaces.Box(low=self.min_speed, high=self.max_speed, shape=(2,), dtype=np.float32),
        #         # "agent_theta": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
        #         # "target_distance": spaces.Box(low=-np.linalg.norm(self.max_location-self.min_location), high=np.linalg.norm(self.max_location-self.min_location), shape=(1,), dtype=np.float32),
        #         # "target_theta": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
        #         "normalized_agent_speed": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        #         "normalized_target_relative_position": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        #         "normalized_relative_target_theta": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        #         "normalized_agent_theta": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        #         "normalized_delta_theta": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        #     }
        # )

        self.observation_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        # self.action_space = spaces.Dict(
        #     {
                
        #         "vel_y": spaces.Box(low=-max_speed, high=max_speed, shape=(1,), dtype=np.float32),
        #     }
        # )
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

    def _get_obs(self):
        normalized_agent_speed = self._agent_speed / self.max_speed
        normalized_target_relative_position = (self._target_location - self._agent_location) / self.map_diagonal_size
        target_theta = math.atan2((self._target_location - self._agent_location)[1],(self._target_location - self._agent_location)[0])
        agent_theta = math.atan2(self._agent_speed[1],self._agent_speed[0])
        normalized_relative_target_theta = target_theta / math.pi
        normalized_agent_theta = agent_theta / math.pi
        delta_theta = target_theta - agent_theta
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta)) 
        normalized_delta_theta = delta_theta / math.pi

        # return {
        #     # "agent_speed": np.concatenate((self._agent_location, self._agent_speed)),            
        #     # "target": self._target_location,            
        #     # "agent_speed": self.normalized_agent_speed,
        #     # "agent_theta": math.atan2(self._agent_speed[1],self._agent_speed[0]),
        #     # "target_distance": np.linalg.norm(self._target_location - self._agent_location),
        #     # "target_theta": math.atan2((self._target_location - self._agent_location)[1],(self._target_location - self._agent_location)[0]),
        #     "normalized_agent_speed": normalized_agent_speed,
        #     "normalized_target_relative_position": normalized_target_relative_position,
        #     "normalized_relative_target_theta": normalized_relative_target_theta,
        #     "normalized_agent_theta": normalized_agent_theta,
        #     "normalized_delta_theta": normalized_delta_theta,
        # }
        return np.array([normalized_agent_speed[0],
                         normalized_agent_speed[1],
                         normalized_target_relative_position[0],
                         normalized_target_relative_position[1],
                         normalized_delta_theta]
                        )

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._target_location - self._agent_location)
        }

    def reset(self, return_info=False, options=None):
        # Choose the UoI's location uniformly at random
        self._agent_location = np.random.uniform(self.min_location, high=self.max_location, size=(2,)).astype(np.float32)
        self._agent_speed = np.random.uniform(self.min_speed, high=self.max_speed, size=(2,)).astype(np.float32)
        self._agent_speed_prev = self._agent_speed        

        # Choose the goal's location uniformly at random
        self._target_location = np.random.uniform(self.min_location, high=self.max_location, size=(2,)).astype(np.float32)        
        # self._target_location = np.zeros(2)

        self._init_target_distance = np.linalg.norm(self._target_location - self._agent_location)
        self._prev_distance = self._init_target_distance 
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):            
        # We use `np.clip` to make sure we don't leave the grid           
        # dx = np.concatenate((action['vel_x'], action['vel_y']) , axis=0) * self.tau
        # action_noise = np.random.normal(np.zeros_like(action), self.max_speed * 0.01)        
        # action += action_noise
        dv = np.clip((action - self._agent_speed_prev)/self.tau, self.min_acceleratoin, self.max_acceleratoin)
        self._agent_speed = self._agent_speed_prev + dv * self.tau
        dx = self._agent_speed * self.tau        
        self._agent_location += dx
        self._agent_speed_prev = self._agent_speed
        clipped_location = np.clip(self._agent_location, self.min_location , self.max_location)

        distance = np.linalg.norm(self._target_location - self._agent_location)      

        reward = 0 
        reward -= 1 / self._init_target_distance        
        reward += 10 * (self._prev_distance - distance)
        delta_theta = math.atan2((self._target_location - self._agent_location)[1],(self._target_location - self._agent_location)[0]) - math.atan2(self._agent_speed[1],self._agent_speed[0])
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))            
        reward -= 0.1 * abs(delta_theta)

        if distance < 0.5:  # An episode is done if the agent has reached the target        
            done = True            
            reward += 1000
        elif (clipped_location != self._agent_location).any():  # An episode is done if the agent has gone out of box            
            done = True            
            reward -= 1500                        
        else:
            done = False
          
        observation = self._get_obs()        
        info = self._get_info()                  
        self.steps += 1      
        
        self._prev_distance = distance                
        return observation, reward, done, info

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
        
        # First we draw the target
        target_render_location = (self._target_location + np.array([self.x_size/2, self.y_size/2])) * pixel_per_meter 
        target_render_location[1] = self.window_size_y - target_render_location[1] 
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                target_render_location,
                (object_render_size, object_render_size),
            ),
        )
        
        
        # Now we draw the agent
        agent_render_location = (self._agent_location + np.array([self.x_size/2, self.y_size/2])) * pixel_per_meter 
        agent_render_location[1] = self.window_size_y - agent_render_location[1] 
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            agent_render_location,
            object_render_size,
        )
        theta_v = math.atan2(-self._agent_speed[1], self._agent_speed[0])
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
