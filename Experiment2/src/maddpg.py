import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sample_env import SampleEnv
from utils import get_average_gradient



class Critic(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(8, 64), # input x1, y1, t1, x2, y2, t2, a1, a2
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)
    

class Actor(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(3, 64),  # input x, y, t
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)
    

class ReplayBuffer:

    def __init__(self, cap, x_dim, a_dim, r_dim) -> None:
        self.state = np.zeros((cap, x_dim), dtype=np.float32)
        self.actions = np.zeros((cap, a_dim), dtype=np.float32)
        self.rewards = np.zeros((cap, r_dim), dtype=np.float32)
        self.next_state = np.zeros((cap, x_dim), dtype=np.float32)
        # we assume a continuous task, thus no 'done' flags here.

        self._index = 0
        self._cap = cap
        self._full = False

    def add(self, x, a, r, x_):
        i = self._index
        self.state[i, :] = x
        self.actions[i, :] = a
        self.rewards[i, :] = r
        self.next_state[i, :] = x_
        self._index += 1
        if self._index == self._cap:
            self._full = True
            self._index = 0

    def sample(self, batch):
        indices = np.random.randint(0, self._index if not self._full else self._cap, (batch, ))
        s = self.state[indices]
        a = self.actions[indices]
        r = self.rewards[indices]
        s_ = self.next_state[indices]
        return s, a, r, s_


class MADDPGAgent:

    def __init__(self) -> None:
        self.critic = [Critic() for _ in range(2)]
        self.actor = [Actor() for _ in range(2)]
        self.target_critic = [Critic() for _ in range(2)]
        self.target_actor = [Actor() for _ in range(2)]

        for i in range(2):
            self.target_actor[i].load_state_dict(self.actor[i].state_dict())
            self.target_critic[i].load_state_dict(self.critic[i].state_dict())
        self.q_opt = [optim.Adam(self.critic[i].parameters(), lr=1e-2) for i in range(2)]  # using smaller learning rate
        self.a_opt = [optim.Adam(self.actor[i].parameters(), lr=1e-3) for i in range(2)]  # using smaller learning rate

    def choose_actions(self, state):
        s_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
        actions = []
        for i in range(2):
            with torch.no_grad():
                a = self.actor[i](s_tensor[:, i * 3: i * 3 + 3])  # state -> obs
                actions.append(a.squeeze(dim=0).item())
        return np.array(actions)
    
    def update(self, x, a, r, x_, ai):
        x_tensor = torch.tensor(x)
        a_tensor = torch.tensor(a)
        r_tensor = torch.tensor(r)
        nx_tensor = torch.tensor(x_)

        next_state_inputs = [nx_tensor]
        
        
        # to construct input for function Qi
        # reference PPT Page43
        for i in range(2):
            with torch.no_grad():
                a_ = self.target_actor[i](nx_tensor[:, i * 3:i * 3 + 3])  # hard coded state -> obs
            next_state_inputs.append(a_)
        
        
        n_sa_tensor = torch.concatenate(next_state_inputs, dim=1)  # NOTICE: check shape
        with torch.no_grad():
            q = r_tensor[:, ai] + 0.95 * self.target_critic[ai](n_sa_tensor).view(-1)

        sa_tensor = torch.concatenate([x_tensor, a_tensor], dim=1)  # NOTICE: check shape
        q_hat = self.critic[ai](sa_tensor).view(-1)

        # print(f'{q_hat.detach().mean()} {q.detach().mean()}')

        # critic loss
        # TODO: add critic_loss
        q_loss_fn = torch.nn.MSELoss()
        q_loss = q_loss_fn(q_hat, q)
        # q_loss = None
        # TODO_END
        self.q_opt[ai].zero_grad()
        q_loss.backward()
        self.q_opt[ai].step()

        # actor loss
        # TODO: add actor_loss
        
        current_state_inputs = [x_tensor]
        
        
        for i in range(2):
            if i == ai:
                new_action = self.actor[ai](x_tensor[:, ai * 3:ai * 3 + 3])
                current_state_inputs.append(new_action)
            else:
                # shape matters
                ith_action = a_tensor[:,i]
                ith_action_shaped = ith_action.view(-1,1)
                current_state_inputs.append(ith_action_shaped)

        # n_sa_tensor = torch.concatenate(next_state_inputs, dim=1)  # NOTICE: check shape
        new_sa_tensor = torch.concatenate(current_state_inputs, dim = 1)
        
        # reference from ddpg.py:
        # new_a_tensor = self._actor(s_tensor)
        # new_sa_tensor = torch.cat([s_tensor, new_a_tensor], dim=1)
        # q = -self._critic(new_sa_tensor).mean()
        
        a_loss = -self.critic[ai](new_sa_tensor).mean()
        #.mean() or .view(-1) ? 
        # a_loss = None
        # TODO_END
        self.q_opt[ai].zero_grad()
        self.a_opt[ai].zero_grad()
        a_loss.backward()
        self.a_opt[ai].step()
        a_grad = get_average_gradient(self.actor[ai])
       
        print(f'Grad: {a_grad}')

        return q_loss.detach().item(), a_loss.detach().item(), a_grad

    def target_update(self):
        with torch.no_grad():
            for ai in range(2):
                for t, o in zip(self.target_actor[ai].parameters(), self.actor[ai].parameters()):
                    t.data = t.data * 0.99 + o.data * 0.01
                for t, o in zip(self.target_critic[ai].parameters(), self.critic[ai].parameters()):
                    t.data = t.data * 0.99 + o.data * 0.01

    def save(self, fn):
        torch.save([self.actor[i].state_dict() for i in range(2)], fn)
    

class MADDPGLearner:

    def __init__(self) -> None:
        self.agent = MADDPGAgent()
        self.env = SampleEnv()
        self.replay_buffer = ReplayBuffer(1000000, 6, 2, 2)
        self.train_steps = 0
        self._sw = SummaryWriter('./logs')  # TODO: for debug

    def train_one_episode(self):
        state = self.env.reset()
        done = False
        total_reward = [0, 0]
        while not done:
            self.train_steps += 1
            actions = self.agent.choose_actions(state)
            exploration_noise = np.random.normal(0, 1, (2, ))
            actions += exploration_noise
            actions = actions.clip(-1, 1)
            assert actions.shape == exploration_noise.shape

            # execute actions
            next_state, rewards, done, _ = self.env.step(actions)
            self.replay_buffer.add(state, actions, rewards, next_state)
            state = next_state
            
            for i in range(2):
                total_reward[i] += rewards[i]

            # update estimators
            if self.train_steps > 10000 and self.train_steps % 100 == 0:
                for i in range(2):
                    x, a, r, x_ = self.replay_buffer.sample(100)
                    ql, al, a_grad = self.agent.update(x, a, r, x_, i)
                    if self.train_steps % 1000 == 0:
                        self._sw.add_scalar(f'train/q_loss_{i}', ql, self.train_steps)
                        self._sw.add_scalar(f'train/a_loss_{i}', al, self.train_steps)
                        self._sw.add_scalar(f'train/a_grad_{i}', a_grad, self.train_steps)

                self.agent.target_update()

        self._sw.add_scalar('reward/train_0', total_reward[0], self.train_steps)
        self._sw.add_scalar('reward/train_1', total_reward[1], self.train_steps)

    def eval_one_episode(self):
        state = self.env.reset()
        done = False
        total_reward = [0, 0]
        while not done:
            actions = self.agent.choose_actions(state)
            next_state, rewards, done, _ = self.env.step(actions)       
            state = next_state
            for i in range(2):
                total_reward[i] += rewards[i]
        
        self._sw.add_scalar('reward/eval_0', total_reward[0], self.train_steps)
        self._sw.add_scalar('reward/eval_1', total_reward[1], self.train_steps)


def main():
    app = MADDPGLearner()
    for episode in range(100000):
        app.train_one_episode()
        if episode % 1000 == 0:
            app.eval_one_episode()
            app.agent.save(f'maddpg/{episode}.th')


if __name__ == '__main__':
    main()
