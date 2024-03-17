import numpy as np


class SampleEnv:

    V = 2 * np.pi / 100
    C = 0.1
    A = np.pi / 4
    DT = 1

    def __init__(self) -> None:
        self.r = 1.
        self.state = np.array([[-1., 0., np.pi / 2], [2., 0., np.pi / 2]])
        self.tot = 0

    def reset(self):

        t = np.random.uniform(-np.pi, np.pi, (2, ))

        t = np.pi / 2 + t
        for i in range(2):
            while t[i] > np.pi:
                t[i] -= np.pi * 2
            while t[i] < -np.pi:
                t[i] += np.pi * 2

        self.state = np.array([  # BUG FIX
            [np.cos(t[0]), np.sin(t[0]), t[0]], 
            [np.cos(t[1]) + 1.0, np.sin(t[1]), t[1]]]
        )
        self.tot = 0
        return self.get_state()

    def step(self, actions):
        assert actions.shape == (2, )  # ACT DIM = 2

        self.state[:, 2] += actions * SampleEnv.A * SampleEnv.DT
        
        for i in range(2):
            while self.state[i, 2] > np.pi:
                self.state[i, 2] -= np.pi * 2
            while self.state[i, 2] < -np.pi:
                self.state[i, 2] += np.pi * 2

        dx = SampleEnv.V * np.cos(self.state[:, 2]) * SampleEnv.DT
        dy = SampleEnv.V * np.sin(self.state[:, 2]) * SampleEnv.DT
        self.state[:, 0] += dx
        self.state[:, 1] += dy
        self.tot += 1
        return self.get_state(), self.get_reward(), self.tot > 100, {}

    def get_state(self):
        return self.state.reshape((-1, )).copy()  # STATE DIM = 6
    
    def get_reward(self):
        r = np.zeros((2, ))
        dist = np.linalg.norm(self.state[0, :2] - self.state[1, :2])
        if dist < 0.1:
            r -= 1. * 0.0  # collision detected

        r0 = np.linalg.norm(self.state[0, :2] - [0, 0])
        r[0] -= abs(r0 - 1) - 1
        r1 = np.linalg.norm(self.state[1, :2] - [1, 0])
        r[1] -= abs(r1 - 1) - 1
        return r


def main():
    env = SampleEnv()
    s = env.reset()
    d = False
    while not d:
        s, r, d, _ = env.step(np.random.uniform(-1, 1, (2, )))
        print(r)


if __name__ == '__main__':
    main()
