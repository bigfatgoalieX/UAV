from fly_circle_discrete import FlyCircle
from dqn import DQNAgent
import time

if __name__ == "__main__":
    env = FlyCircle()
    agent = DQNAgent(env.get_obs_dim(), env.get_action_cnt())
    agent.load(r"dqn\20240310-175853\models\9981.pkl")

    state = env.reset()
    done = False

    while not done:
        env.render()
        time.sleep(0.1)
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
