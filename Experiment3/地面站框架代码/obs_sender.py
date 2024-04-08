import socket
import time
import numpy as np

import struct


def bytes_from_obs(obs):
    # TODO: construct bytes from obs
    bytes_data = struct.pack('ff', obs[0], obs[1])
    print(bytes_data.hex())
    # back_to_float = struct.unpack('ff',bytes_data)
    # print(back_to_float)
    return bytes_data


def send_obs(obs: np.array, ip_addr: str, port: int) -> None:
    # send obs to ip_addr:port
    send_bytes = bytes_from_obs(obs) # TODO: construct bytes from obs
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet & UDP
    sock.sendto(send_bytes, (ip_addr, port))

    # receive from ip_addr:port, and print the msg received
    data, addr = sock.recvfrom(16)
    print(data, addr)



def main():
    np.random.uniform(0, 1, (2, ))
    for step in range(10):
        obs = np.random.uniform(0, 1, (2, ))
        print(obs)
        send_obs(obs, "192.168.31.133", 1234)
        time.sleep(1)


if __name__ == '__main__':
    main()
