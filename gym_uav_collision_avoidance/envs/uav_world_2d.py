import gym
from gym import spaces
import pygame
import numpy as np


class UAVWorld2D(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(self, x_size=100.0, y_size=100.0, agent_num=4, max_speed=10.0, max_acceleration=5.0):
        self.x_size = x_size # size of x dimension
        self.y_size = y_size # size of y dimension
        self.min_location = np.array([-x_size/2.0, -y_size/2.0])
        self.max_location = np.array([x_size/2.0, y_size/2.0])
        self.max_speed = max_speed # Maximum speed of uav
        self.max_acceleration = max_acceleration # Maximum acceleration of uav    
        self.max_window_size = 800  # The size of the PyGame window
        if x_size > y_size:
            self.window_size_x = self.max_window_size
            self.window_size_y = self.max_window_size / x_size * y_size            
        else:
            self.window_size_y = self.max_window_size
            self.window_size_x = self.max_window_size / y_size * x_size        
        

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=self.min_location, high=self.max_location, dtype=np.float64),
                "target": spaces.Box(low=self.min_location, high=self.max_location, dtype=np.float64),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Dict(
            {
                "vel_x": spaces.Box(low=-max_speed, high=max_speed, shape=(1,), dtype=np.float64),
                "vel_y": spaces.Box(low=-max_speed, high=max_speed, shape=(1,), dtype=np.float64),
            }
        )
        # self.action_space = spaces.Box(-max_speed, max_speed, shape=(2,), dtype=np.float64)

        
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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, return_info=False, options=None):
        # Choose the UoI's location uniformly at random
        self._agent_location = np.random.uniform(self.min_location, high=self.max_location, size=(2,)).astype(np.float64)
        
        # Choose the goal's location uniformly at random
        self._target_location = np.random.uniform(self.min_location, high=self.max_location, size=(2,)).astype(np.float64)        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):            
        # We use `np.clip` to make sure we don't leave the grid
        # dt = self.clock.tick(60) * 0.001
        self._agent_location = np.clip(
            self._agent_location + np.concatenate((action['vel_x'], action['vel_y']), axis=0), self.min_location , self.max_location
        )
        # An episode is done iff the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        if self.clock is None:
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
        

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
       

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
