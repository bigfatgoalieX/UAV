import numpy as np
import pygame

VELOCITY = 0.25
MAX_DA = 1
DT = 0.2

class FlyCircle:
    def __init__(self) -> None:
        self.uav_pos = np.zeros(3, dtype=np.float32)
        self.uav_theta = np.zeros(1, dtype=np.float32)

        # circle center and radius
        self.center = np.array([0, -1, 0], dtype=np.float32)
        self.radius = 1

        self.step_cnt = 0

        self.act_space = [-1, -0.5, 0.5, 1]

        self.screen = None

    def get_obs_dim(self):
        return 3

    def get_action_cnt(self):
        return 4

    # reset env state
    def reset(self):
        self.uav_pos = self.center.copy()
        self.uav_pos[1] -= self.radius
        self.uav_theta = np.zeros(1, dtype=np.float32)

        self.step_cnt = 0
        return self.get_obs()

    # apply action in env, return (s_, r, done, __reserved)
    def step(self, action):
        self.step_cnt += 1
        action = self.act_space[action]

        # update uav theta
        self.uav_theta += action * MAX_DA * DT

        if self.uav_theta > np.pi:
            self.uav_theta -= 2 * np.pi
        if self.uav_theta < -np.pi:
            self.uav_theta += 2 * np.pi
        
        # update uav pos
        self.uav_pos[0] += VELOCITY * np.cos(self.uav_theta) * DT
        self.uav_pos[1] += VELOCITY * np.sin(self.uav_theta) * DT

        obs = self.get_obs()
        reward = self.get_reward()
        done = self.get_done()

        return obs, reward, done, ''

    def get_obs(self):
        obs = np.hstack([self.uav_pos[:2] - self.center[:2], self.uav_theta])
        return obs

    def get_reward(self):
        rew = -np.abs(np.linalg.norm(self.uav_pos[:2] - self.center[:2]) - self.radius)
        return rew

    def get_done(self):
        return self.step_cnt >= 200

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.offset = np.array([300, 300]) - self.center[:2]
            self.scale = 50
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        self.screen.fill("white")
        # target circle
        pygame.draw.circle(self.screen, "black", self.center[:2] * self.scale + self.offset, self.scale * self.radius, 1)

        # uav
        pygame.draw.circle(self.screen, "blue", self.uav_pos[:2] * self.scale + self.offset, 5)

        # direction of uav
        line_len = 20
        st = self.uav_pos[:2] * self.scale + self.offset
        ed = st[0] + np.cos(self.uav_theta).item() * line_len, st[1] + np.sin(self.uav_theta).item() * line_len
        pygame.draw.line(self.screen, "black", st, ed)

        pygame.display.flip()
