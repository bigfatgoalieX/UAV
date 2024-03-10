from fly_circle import FlyCircle
from ddpg import DDPGAgent
import time

if __name__ == "__main__":
    env = FlyCircle()
    agent = DDPGAgent(env.get_obs_dim(), env.get_action_dim())
    agent.load("path/to/model")

    state = env.reset()
    done = False

    while not done:
        env.render()
        time.sleep(0.1)
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
