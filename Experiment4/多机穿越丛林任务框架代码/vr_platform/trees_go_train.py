import torch

from algorithm.td3 import TD3Trainer
from util.config import config
from env.trees_go import TreesGo

def main():
    uav_cnt = config.getint("common", "uav_cnt")
    episodes = config.getint("train", "episodes")

    torch.set_num_threads(1)
    env = TreesGo()
    trainer = TD3Trainer(uav_cnt, env)

    for i in range(episodes):
        r = trainer.train_one_episode()
        if i % 200 == 0:
            trainer.save()
            trainer.test_one_episode()


if __name__ == '__main__':
    main()