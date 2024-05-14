import time

from algorithm.td3 import TD3Agent
from util.config import config
from util.communication import SharedBuffer, start_sender, start_receiver
from env.cross_curtain import CrossCurtain
from multiprocessing import Process
from uav.real_uav import RealUav

def main():
    uav_cnt = config.getint("common", "uav_cnt")
    real_uav_cnt = config.getint("common", "real_uav_cnt")

    # uav_0, uav_1, ..., uav_n
    uav_id = [f"uav_{i}" for i in range(uav_cnt)]
    
    send_buffers = {uid: SharedBuffer() for uid in uav_id}
    recv_buffers = {uid: SharedBuffer() for uid in uav_id}

    if real_uav_cnt > 0:
        # send observation to uav
        comm_sender = Process(target=start_sender, args=(send_buffers,), daemon=True)
        comm_sender.start()

        # receive uav position
        comm_receiver = Process(target=start_receiver, args=(recv_buffers, 59999), daemon=True)
        comm_receiver.start()

    real_uavs = [RealUav(recv_buffer=recv_buffers[f"uav_{i}"], send_buffer=send_buffers[f"uav_{i}"]) for i in range(real_uav_cnt)]

    env = CrossCurtain(real_uavs)
    td3agent = TD3Agent(obs_dim=env.get_obs_dim(), act_dim=env.get_action_dim())
    td3agent.load(config.get("evaluate", "model_path"))

    states = env.reset(mode="evaluate")
    env.render()
    input("Press Enter to start...")

    done = {'__all__': False}
    total_rew = {n: 0 for n in states.keys()}

    while not done['__all__']:
        actions = {}
        in_states = []
        enum_seq = list(states.keys())
        for seq in enum_seq:
            in_states.append(states[seq])
        out_actions = td3agent.choose_action(in_states)
        for i, seq in enumerate(enum_seq):
            actions[seq] = out_actions[i]

        next_states, rewards, done, info = env.step(actions)
        for seq in enum_seq:
            total_rew[seq] += rewards[seq]
        env.render()
        if real_uav_cnt == 0:
            time.sleep(0.1)
        states = next_states
        for k, v in done.items():
            if v and k != '__all__':
                del states[k]


if __name__ == '__main__':
    main()