import numpy as np
import pygame
import math

VELOCITY = 0.25
MAX_DA = 1
DT = 0.2

class FlyParabola:
    def __init__(self) -> None:
        self.uav_pos = np.zeros(3, dtype=np.float32)
        self.uav_theta = np.zeros(1, dtype=np.float32)

        # parabola parameters
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.a = -1
        self.b = 0
        self.c = 0
        
        self.focus = np.array([0, -0.25, 0], dtype=np.float32)
        self.directrix = 0.25
        
        self.funcvalue = 0
        
        self.step_cnt = 0

        self.screen = None

    def get_obs_dim(self):
        return 3

    def get_action_dim(self):
        return 1

    # reset env state
    def reset(self):
        self.uav_pos = self.center.copy()
        self.uav_theta = np.zeros(1, dtype=np.float32)

        self.step_cnt = 0
        return self.get_obs()

    # apply action in env, return (s_, r, done, __reserved)
    def step(self, action):
        self.step_cnt += 1

        # update uav theta
        self.uav_theta += action * MAX_DA * DT
        # I don't understand... Is this w*dt?
        
        # make sure theta in [-pi,pi]
        if self.uav_theta > np.pi:
            self.uav_theta -= 2 * np.pi
        if self.uav_theta < -np.pi:
            self.uav_theta += 2 * np.pi
        
        # update uav pos
        self.uav_pos[0] += VELOCITY * np.cos(self.uav_theta) * DT
        self.uav_pos[1] += VELOCITY * np.sin(self.uav_theta) * DT
        
        self.funcvalue = self.get_parabola_value(self.uav_pos[0])

        obs = self.get_obs()
        reward = self.get_reward()
        done = self.get_done()

        return obs, reward, done, ''

    def get_obs(self):
        obs = np.hstack([self.uav_pos[:2] - self.center[:2], self.uav_theta])
        return obs

    def get_reward(self):
        # rew = -np.abs(np.linalg.norm(self.uav_pos[:2] - self.center[:2]) - self.radius)
        distance_to_focus = np.linalg.norm(self.uav_pos[:2] - self.focus[:2])
        distance_to_directrix = np.abs(self.directrix - self.uav_pos[1])
        rew = -np.abs(distance_to_focus - distance_to_directrix)
        return rew

    def get_done(self):
        return self.step_cnt >= 200
    
    def get_parabola_value(self,x):
        return self.a*x**2 + self.b*x + self.c
        

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.offset = np.array([300, 300]) - self.center[:2]
            self.scale = 45
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        self.screen.fill("white")
        
        for x in range(0,600):
            y = self.get_parabola_value(x-300) + 300
            pygame.draw.circle(self.screen, "black", (x,y), 1)
        
        # target circle
        # pygame.draw.circle(self.screen, "black", self.center[:2] * self.scale + self.offset, self.scale * 1, 1)
        
        # target ellipse center (0,-1)
        # pygame.draw.ellipse(self.screen, "black", pygame.Rect(225, 210, 150, 90),1)
        
        #vertical ellipse center (0,0)
        # pygame.draw.ellipse(self.screen, "black", pygame.Rect(255, 225, 90, 150),1)

        # uav
        pygame.draw.circle(self.screen, "blue", self.uav_pos[:2] * self.scale + self.offset, 5)

        # direction of uav
        line_len = 20
        st = self.uav_pos[:2] * self.scale + self.offset
        ed = st[0] + np.cos(self.uav_theta).item() * line_len, st[1] + np.sin(self.uav_theta).item() * line_len
        pygame.draw.line(self.screen, "black", st, ed)

        pygame.display.flip()