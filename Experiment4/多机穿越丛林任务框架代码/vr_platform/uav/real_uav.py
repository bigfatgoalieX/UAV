from util.communication import *
import struct

class RealUav:
    
    def __init__(self, recv_buffer: SharedBuffer, send_buffer: SharedBuffer) -> None:
        self.recv_buffer = recv_buffer
        self.send_buffer = send_buffer
        self.state = {}

    def set_state(self, state):
        pass

    def get_state(self):
        data = b""
        while len(data) == 0:
            data = self.recv_buffer.read()
        self.state = pickle.loads(data)
        return self.state

    def take_action(self, obs, action):
        if self.state["stop"]:
            return
        p_k, vel, z = 1.5, 0.25, self.state["pos"][2]
        data = struct.pack("f" * len(obs) + "fff", *obs, p_k, vel, z)
        self.send_buffer.write_with_v(data)