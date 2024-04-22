from fly_circle import FlyCircle
from fly_ellipse import FlyEllipse
from fly_parabola import FlyParabola
from fly_moving_circle import FlyMovingCircle
from td3 import TD3Agent
import time

if __name__ == "__main__":
    env = FlyMovingCircle()
    agent = TD3Agent(env.get_obs_dim(), env.get_action_dim())
    agent.load(r"td3\20240422-142146_movingcircle\models\981.pkl")

    state = env.reset()
    done = False

    while not done:
        env.render()
        time.sleep(0.1)
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
