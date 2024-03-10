from fly_circle_discrete import FlyCircle
from double_dqn import DDQNAgent
import time

if __name__ == "__main__":
    env = FlyCircle()
    agent = DDQNAgent(env.get_obs_dim(), env.get_action_cnt())
    agent.load(r"ddqn\20240311-000815_better\models\9981.pkl")

    state = env.reset()
    done = False

    while not done:
        env.render()
        time.sleep(0.1)
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
