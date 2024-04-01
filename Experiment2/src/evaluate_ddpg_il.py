from sample_env import SampleEnv
from ddpg_il import DDPGAgent
import torch
import numpy as np

if __name__ == "__main__":
    env = SampleEnv()
    agent = [DDPGAgent(6, 1) for _ in range(2)]

    state_dicts = torch.load(r"ddpg_il\20240401-162925\models\981.pkl")
    for i in range(2):
        agent[i].load_state_dict(state_dicts[i])

    state = env.reset()
    done = False

    tot_reward = 0
    while not done:
        action = []
        for i in range(2):
            action.append(agent[i].choose_action(state).item())
        next_state, reward, done, info = env.step(np.array(action))
        state = next_state
        tot_reward += reward.sum()
    print(tot_reward)
