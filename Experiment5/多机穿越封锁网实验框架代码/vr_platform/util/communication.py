from socketserver import BaseRequestHandler, UDPServer
from socket import socket, AF_INET, SOCK_DGRAM
from multiprocessing.sharedctypes import Value, Array
from multiprocessing import Semaphore
import concurrent.futures
import pickle

from util.config import config

class SharedBuffer:

    def __init__(self, max_size: int = 65530) -> None:
        self._shared_array = Array('c', max_size)  
        self._max_size = max_size
        self._size = Value('i', 0)
        self._lock = Value('i', 0)
        self._sem = Semaphore(0)

    def write(self, data: bytes) -> None:
        self._lock.acquire()

        size = len(data)
        assert size <= self._max_size
        self._shared_array[:size] = data
        self._size.value = size

        self._lock.release()

    def write_with_v(self, data: bytes) -> None:
        self._lock.acquire()

        size = len(data)
        assert size <= self._max_size
        self._shared_array[:size] = data
        self._size.value = size

        self._lock.release()
        self._sem.release()
    
    def read(self) -> bytes:
        self._lock.acquire()
        
        data = self._shared_array[:self._size.value]

        self._lock.release()
        return data
    
    def read_with_p(self) -> bytes:
        self._sem.acquire()
        self._lock.acquire()
        
        data = self._shared_array[:self._size.value]

        self._lock.release()
        return data

class CommHandler(BaseRequestHandler):

    recv_buffers = None
    recv_sn = None

    def handle(self):
        msg, sk = self.request
        udp_sn, uav_id, data = pickle.loads(msg)
        if udp_sn >= CommHandler.recv_sn[uav_id]:
            CommHandler.recv_sn[uav_id] = udp_sn + 1
            CommHandler.recv_buffers[uav_id].write(data)


class CommServer:

    def __init__(self, send_buffers: dict = {}, recv_buffers: dict = {}, port: int = 0) -> None:
        # xxx_buffers -> {uav_id: buffer...}
        self._send_buffers = send_buffers
        self._recv_buffers = recv_buffers
        self._sockets = {uav_id: socket(AF_INET, SOCK_DGRAM) for uav_id in self._send_buffers}

        real_uav_cnt = config.getint("common", "real_uav_cnt")
        self.hosts = {f"uav_{i}": config.get(f"uav_{i}", "ip") for i in range(real_uav_cnt)}
        self.ports = {f"uav_{i}": config.getint(f"uav_{i}", "port") for i in range(real_uav_cnt)}

        CommHandler.recv_buffers = recv_buffers
        CommHandler.recv_sn = {f'uav_{i}': 0 for i in range(real_uav_cnt)}
        
        if port > 0:
            self._udp_serv = UDPServer(('', port), CommHandler)

        self._tp_executor = concurrent.futures.ThreadPoolExecutor(max_workers=real_uav_cnt)
        
    def run_sender(self):
        while True:
            futures = [self._tp_executor.submit(self._send_to, uav_id) for uav_id in self.hosts]
            concurrent.futures.wait(futures)
                
    def run_receiver(self):
        futures = [self._tp_executor.submit(self._udp_serv.serve_forever) for _ in range(10)]
        concurrent.futures.wait(futures)

    def _send_to(self, uav_id: str):
        data = self._send_buffers[uav_id].read_with_p()
        try:
            self._sockets[uav_id].sendto(data, (self.hosts[uav_id], self.ports[uav_id]))
        except Exception as e:
            print(e)


def start_sender(send_buffers):
    comm_server = CommServer(send_buffers=send_buffers)
    comm_server.run_sender()


def start_receiver(recv_buffers, port):
    comm_server = CommServer(recv_buffers=recv_buffers, port=port)
    comm_server.run_receiver()
