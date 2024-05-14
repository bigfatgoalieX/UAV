import numpy as np
import time
import pygame

from uav.sim_uav import SimUav
from util.config import config


class CrossCurtain:

    def __init__(self, real_uavs: list = []):
        self.uav_cnt = config.getint("common", "uav_cnt")
        self.real_uav_cnt = config.getint("common", "real_uav_cnt")
        self.dt = config.getfloat("uav", "dt")
        assert self.real_uav_cnt == len(real_uavs)

        self.uav_list = real_uavs
        for _ in range(self.uav_cnt - self.real_uav_cnt):
            self.uav_list.append(SimUav())
        self.uavs = {f"uav_{i}": self.uav_list[i] for i in range(self.uav_cnt)}

        self._pos = np.zeros((self.uav_cnt, 3))
        self._tar = np.zeros((self.uav_cnt, 2))
        self._theta = np.zeros((self.uav_cnt,))
        self._succ = np.zeros(self.uav_cnt, np.bool8)
        self._cur_center = np.zeros((2, 3))
        self._step = 0

        self._prev_pos = None
        self._passed = np.zeros(self.uav_cnt, np.bool8)

        self.screen = None

    def get_obs_dim(self):
        return 11
    
    def get_action_dim(self):
        return 2
    
    def update_state(self):
        for i, uav in enumerate(self.uavs.values()):
            state = uav.get_state()
            self._pos[i] = state['pos']
            self._theta[i] = state['theta']

    def reset(self, mode="train"):
        self._pos = np.random.uniform([-1.5, -2.5, 0.55], [1.5, -1.5, 0.65], (self.uav_cnt, 3))
        self._theta = np.random.uniform(np.pi * 1 / 4, np.pi * 3 / 4, self.uav_cnt)

        for i, uav in enumerate(self.uavs.values()):
            uav.set_state({'pos': self._pos[i], 'theta': self._theta[i]})

        self.update_state()

        self._tar = np.copy(self._pos[:, :2])
        self._tar[:, 1] += 4
        self._succ[:] = False
        self._passed[:] = False

        center_y = 0
        self._cur_center[0] = np.random.uniform([-1.0, center_y, 0.8], [0, center_y, 1.5])
        self._cur_center[1] = np.random.uniform([0, center_y, 0.8], [1, center_y, 1.5])

        if self.real_uav_cnt > 0:
            self._pos[0] = np.array([-1.5, -2.5, 0.8])
            self._pos[1] = np.array([1.5, -2.5, 0.8])
            self._pos[2] = np.array([-1.5, -1.5, 0.8])
            self._pos[3] = np.array([1.5, -1.5, 0.8])
            self._tar = np.copy(self._pos[:, :2])
            self._tar[:, 1] += 4
            self._theta[:] = np.pi / 2
            for i, uav in enumerate(self.uavs.values()):
                uav.set_state({'pos': self._pos[i], 'theta': self._theta[i]})
            self._cur_center[0] = np.array([-1, 0, 0.8])
            self._cur_center[1] = np.array([1, 0, 1.3])

        self._step = 0
        self._prev_pos = None
        return self.get_obs()

    def step(self, actions):
        self._step += 1
        self.update_state()
        self._prev_pos = np.copy(self._pos)

        obs = self.get_obs()
        for uid, action in actions.items():
            i = int(uid.split("_")[1])
            if self._succ[i]:
                continue
            uav = self.uavs[uid]
            uav.take_action(obs[uid], action)

        if self.real_uav_cnt > 0:
            time.sleep(self.dt)
        self.update_state()
        obs = self.get_obs()
        rew = self.get_rew(obs)
        done = self.get_done()
        return obs, rew, done, {}

    def get_obs(self):
        obs = {}
        for i in range(self.uav_cnt):
            # FIND THE CLOSEST UAV in the XOY plane
            dist = np.linalg.norm(self._pos[:, :2] - self._pos[i, :2], axis=1)
            dist[self._succ] = 1e9
            dist[i] = 1e9
            c_index = np.argmin(dist)

            rel_pos_to_closest_uav = self._pos[i, :2] - self._pos[c_index, :2]
            if not self._succ[i] and np.sum(self._succ) == self.uav_cnt - 1:
                rel_pos_to_closest_uav = self._pos[i, :2] - np.zeros(2)

            obs[f'uav_{i}'] = np.hstack([
                self._pos[i, :2] - self._tar[i],
                self._theta[i],

                self._pos[i] - self._cur_center[0],
                self._pos[i] - self._cur_center[1],

                rel_pos_to_closest_uav,
            ])
        return obs

    def get_rew(self, obs):
        rew = {}
        prev_dist = np.linalg.norm(self._prev_pos[:, :2] - self._tar[:, :2], axis=1)
        now_dist = np.linalg.norm(self._pos[:, :2] - self._tar[:, :2], axis=1)

        for i in range(self.uav_cnt):
            r = 0
            r += -(now_dist[i] - prev_dist[i])  # DIST BASED REW

            dist2uav = np.linalg.norm(obs[f'uav_{i}'][9:11])
            if dist2uav < 0.2:  # COLLISION RISK
                r += -2

            dist2tar = np.linalg.norm(self._pos[i, :2] - self._tar[i, :2])
            if dist2tar < 0.2:  # REACH TARGET
                self._succ[i] = True
                r += 20

            if -0.15 < self._pos[i, 1] < 0.15:
                if not self.in_the_hole(self._pos[i]):  # COLLISION WITH CURTAIN
                    r += -2

            rew[f'uav_{i}'] = r
        return rew
    
    def in_the_hole(self, pos):
        px = pos[0]
        pz = pos[2]

        c_w = 0.4
        for i in range(2):
            cx = self._cur_center[i, 0]
            cz = self._cur_center[i, 2]

            if cx - c_w / 2 < px < cx + c_w / 2 and cz - c_w / 2 < pz < cz + c_w / 2:  # IN THE HOLE
                return True
        return False

    def get_done(self):
        done = {f"uav_{i}": self._succ[i] for i in range(self.uav_cnt)}
        done["__all__"] = self._step >= 100 or all(done.values())
        return done
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.offset = np.array([300, 300])
            self.scale = 75
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        self.screen.fill("white")

        for i in range(2):
            st = self._cur_center[i, :2].copy()
            ed = self._cur_center[i, :2].copy()
            st[0] -= 0.2
            ed[0] += 0.2
            st = st * self.scale + self.offset
            ed = ed * self.scale + self.offset
            pygame.draw.line(self.screen, "black", st, ed)

        for i in range(self.uav_cnt):
            pos, theta = self._pos[i, :2], self._theta[i]

            color = "blue"
            if i < self.real_uav_cnt:
                color = "purple"
            if -0.15 < self._pos[i, 1] < 0.15 and not self.in_the_hole(self._pos[i]):
                color = "red"

            pygame.draw.circle(self.screen, color, pos * self.scale + self.offset, 5)

            line_len = 20
            st = pos * self.scale + self.offset
            ed = st[0] + np.cos(theta).item() * line_len, st[1] + np.sin(theta).item() * line_len
            pygame.draw.line(self.screen, "black", st, ed)

            pygame.draw.circle(self.screen, "red", self._tar[i] * self.scale + self.offset, 5)

        pygame.display.flip()
