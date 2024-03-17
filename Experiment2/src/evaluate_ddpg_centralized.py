from sample_env import SampleEnv
from ddpg_centralized import DDPGAgent

if __name__ == "__main__":
    env = SampleEnv()
    agent = DDPGAgent(6, 2)
    agent.load(r"ddpg_centralized\20240310-175318\models\981.pkl")

    state = env.reset()
    done = False

    tot_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        tot_reward += reward.sum()
    print(tot_reward)
