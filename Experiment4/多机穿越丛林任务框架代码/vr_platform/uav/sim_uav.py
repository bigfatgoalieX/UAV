import numpy as np
from util.config import config

VELOCITY = config.getfloat("uav", "velocity")
MAX_DA = config.getfloat("uav", "max_da")
MAX_DZ = config.getfloat("uav", "max_dz")
DT = config.getfloat("uav", "dt")


class SimUav:
    
    def __init__(self) -> None:
        self.pos = np.zeros(3, dtype=np.float32)
        self.theta = np.zeros(1, dtype=np.float32)

    def set_state(self, state):
        self.pos = state['pos'].copy()
        self.theta = state['theta'].copy()

    def get_state(self):
        state = {}
        state['pos'] = self.pos.copy()
        state['theta'] = self.theta.copy()
        return state

    def take_action(self, obs, action):
        self.theta += action[0] * MAX_DA * DT

        if self.theta > np.pi:
            self.theta -= 2 * np.pi
        if self.theta < -np.pi:
            self.theta += 2 * np.pi
        
        self.pos[0] += VELOCITY * np.cos(self.theta) * DT
        self.pos[1] += VELOCITY * np.sin(self.theta) * DT
        self.pos[2] += action[1] * MAX_DZ * DT
