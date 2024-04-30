import math
import time
from math import sin, cos
import pygame

import numpy as np
from scipy.spatial import KDTree
from uav.sim_uav import SimUav
from util.config import config

N_DIV = 11
MAX_RANGE = 1
V_ANGLE = 150 / 180 * np.pi
LAMBDA = 0.99


def line_intersect_circle(p, lsp, esp):
    # p is the circle parameter, lsp and lep is the two end of the line
    x0, y0, r0 = p
    x1, y1 = lsp
    x2, y2 = esp
    if r0 == 0:
        return [[x1, y1]]
    if abs(x1 - x2) < 1e-8:
        if abs(r0) >= abs(x1 - x0):
            p1 = x1, y0 - math.sqrt(r0 ** 2 - (x1 - x0) ** 2)
            p2 = x1, y0 + math.sqrt(r0 ** 2 - (x1 - x0) ** 2)
            inp = [p1, p2]
        else:
            inp = []
    else:
        k = (y1 - y2) / (x1 - x2)
        b0 = y1 - k * x1
        a = k ** 2 + 1
        b = 2 * k * (b0 - y0) - 2 * x0
        c = (b0 - y0) ** 2 + x0 ** 2 - r0 ** 2
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            p1x = (-b - math.sqrt(delta)) / (2 * a)
            p2x = (-b + math.sqrt(delta)) / (2 * a)
            p1y = k * p1x + b0
            p2y = k * p2x + b0
            inp = [[p1x, p1y], [p2x, p2y]]
        else:
            inp = []
    return inp


def limit_theta(t):
    while t > np.pi:
        t -= 2 * np.pi

    while t < -np.pi:
        t += 2 * np.pi
    return t


def split_theta(center_theta, theta_range, n):
    assert n % 2 == 1
    theta_cell = theta_range / (n - 1)
    res = [center_theta]
    for i in range(1, (n - 1) // 2 + 1):
        res.append(limit_theta(center_theta + theta_cell * i))
    for i in range(1, (n - 1) // 2 + 1):
        res.append(limit_theta(center_theta - theta_cell * i))
    # res.sort(), dont sort
    return res


def get_min_dist_on_path(x, y, theta, cx, cy, r):
    s = sin(theta)
    c = cos(theta)
    res = line_intersect_circle((cx, cy, r), (x, y), (x + c, y + s))
    min_dist = 1e9
    for p in res:
        v1 = np.array(p) - np.array([x, y])
        v2 = np.array([c, s])
        d = np.dot(v1, v2) / np.linalg.norm(v1)
        assert abs(d - 1) < 1e-8 or abs(d + 1) < 1e-8
        if d > 0:
            dist = np.linalg.norm(np.array(p) - np.array([x, y]))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def get_dis_from_lidar(x, y, theta, circles):
    result = []
    theta_s = split_theta(theta, V_ANGLE, N_DIV)
    for t in theta_s:
        min_dist = 1e9
        for c in circles:
            dist = get_min_dist_on_path(x, y, t, c[0], c[1], c[2])
            min_dist = min(min_dist, dist)
        if min_dist > MAX_RANGE:
            min_dist = -1
        result.append(min_dist)
    return result


class TreesGo:

    UAV_OBS_DIS = 1.0

    CRASH_DIS = 0.25
    TARGET_DIS = 1.0
    U_COLLISION_DIS = 0.5

    MAX_STEP = 300

    def __init__(self, real_uavs=[], random_action_noise_std=0.0, alpha=10.0, beta=10.0):
        self.uav_list = real_uavs
        self.uav_cnt = config.getint("common", "uav_cnt")
        self.real_uav_cnt = config.getint("common", "real_uav_cnt")
        self.dt = config.getfloat("uav", "dt")

        assert self.uav_cnt <= 4
        # 注意在trees_go.ini中修改真机的数量
        assert self.real_uav_cnt == len(real_uavs)

        for _ in range(self.uav_cnt - self.real_uav_cnt):
            self.uav_list.append(SimUav())

        self.uavs = {f"uav_{i}": self.uav_list[i] for i in range(self.uav_cnt)}

        self._pos = np.zeros((self.uav_cnt, 2), dtype=np.float32)
        self._tar = np.zeros((self.uav_cnt, 2), dtype=np.float32)
        self._theta = np.zeros((self.uav_cnt,), dtype=np.float32)

        self._dist = np.zeros((self.uav_cnt, self.uav_cnt), dtype=np.float32)

        self._obstacles = None
        self._kd = None
        self._step = 0

        self._collision_this_round = {}
        self._prev_pos = None

        self._random_action_noise_std = random_action_noise_std
        self._render = None

        self.obs_dim = 14 + 6 + 2
        self.act_dim = 1

        self.alpha = alpha
        self.beta = beta

        self._done = np.zeros((self.uav_cnt, ), dtype=np.bool8)

        self.stat = None

        self.screen = None

    def get_obs_dim(self):
        return self.obs_dim

    def get_action_dim(self):
        return self.act_dim
    
    def update_state(self):
        for i, uav in enumerate(self.uavs.values()):
            state = uav.get_state()
            self._pos[i] = state['pos'][:2]
            self._theta[i] = state['theta']

    def _update_dist(self):
        for i in range(self.uav_cnt):
            self._dist[i, :] = np.linalg.norm(self._pos - self._pos[i], axis=1)
            self._dist[i, i] = 1e9

    def _limit_theta(self):
        self._theta[self._theta > np.pi] -= 2 * np.pi
        self._theta[self._theta < -np.pi] += 2 * np.pi

    def get_obs(self):
        obs = {}
        for i in range(self.uav_cnt):
            name = f'uav_{i}'
            if self._done[i]:
                continue
            pos = self._pos[i]
            theta = self._theta[i]
            # CHECK: query all trees within xx meters  (0.25 is the largest radius of all trees)
            points = self._kd.query_ball_point(pos, 0.25 + MAX_RANGE)
            circles = self._obstacles[points]
            obstacle_dist = get_dis_from_lidar(pos[0], pos[1], theta, circles)
            
            # CHECK: query all uavs within 1.0 meters
            lst = np.argsort(self._dist[i, :])
            nearby_uavs = []
            for j in range(2):
                if self._dist[i, lst[j]] < TreesGo.UAV_OBS_DIS:
                    nearby_uavs.append(self._pos[lst[j], 0])
                    nearby_uavs.append(self._pos[lst[j], 1])
                    nearby_uavs.append(self._theta[lst[j]])
                else:
                    nearby_uavs.extend([2., 2., 0])
            
            this_uav_obs = np.array([pos[0], pos[1], self._tar[i, 0] - pos[0], self._tar[i, 1] - pos[1], theta, *obstacle_dist, *nearby_uavs])
            obs[name] = this_uav_obs
        return obs

    def get_rew(self):
        prev_dist = np.linalg.norm(self._prev_pos - self._tar, axis=1)
        next_dist = np.linalg.norm(self._pos - self._tar, axis=1)

        dt = -(next_dist - prev_dist)
        rew = {}
        for i in range(self.uav_cnt):
            name = f'uav_{i}'
            if self._done[i]:
                continue
            # dist based reward
            rew[name] = dt[i]

            # obstacle avoidance based reward
            pts = self._kd.query_ball_point(self._pos[i], TreesGo.CRASH_DIS)
            for p in pts:
                radius = self._obstacles[p][2]
                dist = np.linalg.norm(np.array(self._obstacles[p, :2]) - self._pos[i])
                if dist < radius:
                    rew[name] -= 1.0 * self.alpha  # COLLISION PENALTY
                    self.stat['c1'][i] += 1

            # inter-UAV collision avoidance
            for j in range(self.uav_cnt):
                if self._dist[i, j] < TreesGo.U_COLLISION_DIS and not self._done[j]:
                    rew[name] -= 1.0 * self.beta
                    self.stat['c2'][i] += 1
            
            rew[name] -= 0.1  # TIME

            # out-of-bound penalty
            px, py = self._pos[i]
            if px < -2.39 or px > 1.59 or py < -6.79 or py > 1.22:
                rew[name] -= 1.
            
            if np.linalg.norm(self._pos[i] - self._tar[i]) < TreesGo.TARGET_DIS:  # reach!!!
                rew[name] += 50
        return rew

    def get_done(self):
        done = {}
        for i in range(self.uav_cnt):
            name = f'uav_{i}'
            if self._done[i]:
                continue
            done[name] = False

            if np.linalg.norm(self._pos[i] - self._tar[i]) < TreesGo.TARGET_DIS:  # reach!!!
                done[name] = True
                self._done[i] = True
                if self.stat['ta'][i] > TreesGo.MAX_STEP:
                    self.stat['ta'][i] = self._step

        done['__all__'] = all(done.values()) or self._step > TreesGo.MAX_STEP
        return done

    def reset(self, mode="train"):
        self._done = np.zeros((self.uav_cnt, ), np.bool8)
        
        number_of_trees = 10

        self._obstacles = np.random.uniform([-2.39, -5.0, TreesGo.CRASH_DIS], [1.59, 0.5, TreesGo.CRASH_DIS], (number_of_trees, 3)) # np.array(random_trees)
        self._kd = KDTree(self._obstacles[:, :2])  # add all center of trees into kd-tree

        pos = [[1.6, 1.2], [-1.1, 1.2], [0.1, 1.2], [-2.3, 1.2]]
        tar = [[1.0, -6.0], [0.0, -6.0], [-1.0, -6.0], [-2.0, -6.0]]
        # np.random.shuffle(pos)
        np.random.shuffle(tar)

        self._pos = np.array(pos[:self.uav_cnt])
        self._tar = np.array(tar[:self.uav_cnt])
        self._theta = np.random.uniform(-np.pi * 3 / 4, -np.pi / 4, self.uav_cnt)

        self._step = 0
        self._prev_pos = np.copy(self._pos)

        self.stat = {
            'c1': [0 for _ in range(self.uav_cnt)],                    # count of collisions between uav and obstacle
            'c2': [0 for _ in range(self.uav_cnt)],                    # count of collisions between uav and uav
            'ta': [TreesGo.MAX_STEP + 1 for _ in range(self.uav_cnt)]  # step count before uav reaches target
        }

        for i, uav in enumerate(self.uavs.values()):
            uav.set_state({'pos': np.hstack([self._pos[i], 0]), 'theta': self._theta[i]})
        self.update_state()

        if mode == "evaluate":
            self._obstacles = np.array([
                [0.59, 0.22, TreesGo.CRASH_DIS],
                [-1.41, -0.78, TreesGo.CRASH_DIS],
                [2.58, -0.78, TreesGo.CRASH_DIS],
                [0.58, -1.80, TreesGo.CRASH_DIS],
                [-2.43, -2.77, TreesGo.CRASH_DIS],
                [-0.42, -2.78, TreesGo.CRASH_DIS],
                [1.59, -2.78, TreesGo.CRASH_DIS],
                [-1.41, -3.79, TreesGo.CRASH_DIS],
                [0.58, -4.80, TreesGo.CRASH_DIS],
                [-1.42, -4.80, TreesGo.CRASH_DIS],
            ])
            self._kd = KDTree(self._obstacles[:, :2])  # add all center of trees into kd-tree
            self._tar = np.array([[1.0, -6.0], [0.0, -6.0], [-1.0, -6.0], [-2.0, -6.0]])

        return self.get_obs()

    def step(self, actions):
        self._step += 1
        self._prev_pos = np.copy(self._pos)
        u = np.zeros((self.uav_cnt, ))

        d_stop = []
        for i in range(self.uav_cnt):
            name = f'uav_{i}'
            d = self._done[i]
            if name in actions:
                u[i] = actions[name]
            else:
                d = True
            d_stop.append(d)

        # state transition
        self.update_state()
        obs = self.get_obs()
        for i, uav in enumerate(self.uavs.values()):
            if not self._done[i]:
                uav.take_action(obs[f'uav_{i}'], [u[i], 0])

        # waiting for real uav to fly...
        if self.real_uav_cnt > 0:
            time.sleep(self.dt)
        self.update_state()

        self._update_dist()
        obs = self.get_obs()
        rew = self.get_rew()
        done = self.get_done()

        return obs, rew, done, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.offset = np.array([300, 500])
            self.scale = 75
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        self.screen.fill("white")

        for i in range(self._obstacles.shape[0]):
            pos = self._obstacles[i, :2]
            pygame.draw.circle(self.screen, "green", pos * self.scale + self.offset, 5)

        for i in range(self.uav_cnt):
            pos, theta = self._pos[i, :2], self._theta[i]
            color = "blue"
            if i < self.real_uav_cnt:
                color = "purple"
            pygame.draw.circle(self.screen, color, pos * self.scale + self.offset, 5)

            line_len = 20
            st = pos * self.scale + self.offset
            ed = st[0] + np.cos(theta).item() * line_len, st[1] + np.sin(theta).item() * line_len
            pygame.draw.line(self.screen, "black", st, ed)

            pygame.draw.circle(self.screen, "red", self._tar[i] * self.scale + self.offset, 5)

        pygame.display.flip()
