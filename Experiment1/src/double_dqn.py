import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from fly_circle_discrete import FlyCircle
from datetime import datetime

LR = 1e-4
GAMMA = 0.95
EPSILON = 0.1
BATCH_SIZE = 32
START_UPD_SAMPLES = 500
TAU = 50

MAIN_FOLDER = "ddqn/" + datetime.now().strftime("%Y%m%d-%H%M%S")


class QNet(torch.nn.Module):

    def __init__(self, state_dim, action_count):
        super().__init__()
        self.layer_0 = torch.nn.Linear(state_dim, 64)
        self.layer_1 = torch.nn.Linear(64, 32)
        self.layer_2 = torch.nn.Linear(32, action_count)

    def forward(self, x):
        x = torch.relu(self.layer_0(x))
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class ReplayBuffer:

    def __init__(self, cap, state_shape):
        self._state = np.zeros((cap, state_shape), dtype=np.float32)
        self._action = np.zeros((cap,), dtype=np.int32)
        self._reward = np.zeros((cap,), dtype=np.float32)
        self._next_state = np.zeros((cap, state_shape), np.float32)
        self._done = np.zeros((cap,), dtype=np.bool_)
        self._index = 0
        self._cap = cap
        self._full = False

    # add transition
    def add(self, s, a, r, s_, d):
        self._state[self._index] = s
        self._action[self._index] = a
        self._reward[self._index] = r
        self._next_state[self._index] = s_
        self._done[self._index] = d
        self._index += 1
        if self._index == self._cap:
            self._full = True
            self._index = 0

    # sample transitions from replay buffer
    def sample(self, batch):
        if self._full:
            indices = np.random.randint(0, self._cap, (batch,))
        else:
            indices = np.random.randint(0, self._index, (batch,))
        return (self._state[indices], self._action[indices], self._reward[indices],
                self._next_state[indices], self._done[indices])


class DDQNAgent:

    def __init__(self, obs_dim, act_cnt, sw=None):
        self._qnet = QNet(obs_dim, act_cnt)
        self._target_qnet = QNet(obs_dim, act_cnt)
        self._target_qnet.load_state_dict(self._qnet.state_dict())

        self._qnet_opt = torch.optim.Adam(self._qnet.parameters(), lr=LR)

        self._act_cnt = act_cnt
        self._obs_dim = obs_dim
        self._step = 0
        self._sw = sw

    # deterministic policy
    def choose_action(self, obs):
        with torch.no_grad():
            s = torch.from_numpy(obs).float()
            return torch.argmax(self._qnet(s)).numpy().item()
        
    # epsilon-greedy policy
    def choose_action_with_exploration(self, obs):
        with torch.no_grad():
            seed = np.random.uniform(0, 1)
            if seed < EPSILON:
                return np.random.randint(0, self._act_cnt)
            else:
                return self.choose_action(obs)

    # update network parameter
    def update(self, s, a, r, s_, d):
        self._step += 1
        s = torch.tensor(s).float()
        a = torch.tensor(a).long()
        r = torch.tensor(r).float()
        s_ = torch.tensor(s_).float()

        with torch.no_grad():
            # TODO: generate target of Double DQN
            target = None

        # update qnet parameter
        q_s = self._qnet(s)
        loss_function = torch.nn.MSELoss()
        loss = loss_function(q_s, target)
        self._qnet_opt.zero_grad()
        loss.backward()
        self._qnet_opt.step()

        # update target qnet parameter
        if self._step % TAU == 0:
            self._target_qnet.load_state_dict(self._qnet.state_dict())

        if self._step % 500 == 0:
            self._sw.add_scalar('loss/q', loss, self._step)

    def load(self, path):
        self._qnet.load_state_dict(torch.load(path))
    
    def state_dict(self):
        return self._qnet.state_dict()


class DDQNTrainer:

    def __init__(self, env):
        self._env = env
        self._obs_dim = self._env.get_obs_dim()
        self._act_cnt = self._env.get_action_cnt()
        self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs/trainer')
        self._agent = DDQNAgent(self._obs_dim, self._act_cnt, self._sw)
        self._replay_buffer = ReplayBuffer(500000, self._obs_dim)
        self._now_ep = 0
        self._step = 0

    def train_one_episode(self):
        self._now_ep += 1

        state = self._env.reset()
        done = False
        total_rew = 0

        while not done:
            self._step += 1

            # collect transition
            action = self._agent.choose_action_with_exploration(state)
            next_state, reward, done, info = self._env.step(action)
            self._replay_buffer.add(state, action, reward, next_state, done)

            # update network parameter
            if self._step % 50 == 0 and self._step > START_UPD_SAMPLES:
                for _ in range(20):
                    s, a, r, s_, d = self._replay_buffer.sample(BATCH_SIZE)
                    self._agent.update(s, a, r, s_, d)

            total_rew += reward
            state = next_state
        if self._now_ep % 20 == 0:
            self._sw.add_scalar(f'train_rew', total_rew, self._now_ep)
        return total_rew
    
    def test_one_episode(self):
        state = self._env.reset()
        done = False
        total_rew = 0

        while not done:
            action = self._agent.choose_action_with_exploration(state)
            next_state, reward, done, info = self._env.step(action)

            total_rew += reward
            state = next_state

        self._sw.add_scalar(f'test_rew', total_rew, self._now_ep)
        return total_rew
    
    def save(self):
        path = f'./{MAIN_FOLDER}/models'
        if not os.path.exists(path):
            os.makedirs(path)
        save_pth = path + '/' + f'{self._now_ep}.pkl'
        torch.save(self._agent.state_dict(), save_pth)


if __name__ == "__main__":
    torch.set_num_threads(1)
    env = FlyCircle()
    trainer = DDQNTrainer(env)

    for i in range(10000):
        r = trainer.train_one_episode()
        if i % 20 == 0:
            trainer.save()
            trainer.test_one_episode()
