import pickle
import socket
import struct
import time

import torch
from torch.utils.data import DataLoader

# from federated.models import *
# from ..clients import all_arch
from ..utils import clear_parameter,seconds_to_hms


class BaseServer:
    def __init__(
            self,
            ip: str,
            port: int,
            global_epoch: int, # 全局迭代轮次
            n_clients: int,
            model: str,
            data: DataLoader,
            n_classes: int,
            device: str
    ):
        self.ip = ip
        self.port = port
        self.global_epoch = global_epoch  # 全局epoch
        self.n_clients = n_clients  # 客户端个数
        self.data = data  # 测试集
        self.device = device
        from ..register import all_arch
        self.model = all_arch[model](num_classes=n_classes).to(self.device)  # 全局模型
        self.cnt = 0
        self.server_socket = socket.socket()
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(self.n_clients)
        self.clients_socket = []
        self.para_cache = []
        self.round = 1
        self.total = 0
        
        self.now_time=None
        self.start_time=time.time()

    def first_push(self):
        while self.cnt < self.n_clients:
            client_socket, address = self.server_socket.accept()
            self.clients_socket.append(client_socket)
            self.cnt += 1
            #准备第一轮包
        for client_socket in self.clients_socket:
            client_socket.sendall(pickle.dumps(self.model.state_dict()))
            client_socket.sendall(b'stop!')
            client_socket.close()
        self.cnt = 0
        self.clients_socket.clear()

    def run(self):
        print(f"SERVER@{self.ip}:{self.port} INFO: Start!")
        self.first_push()
        for _ in range(self.global_epoch):
            self.pull()
            self.aggregate()
            self.push()
            self.clients_socket.clear()
            self.para_cache.clear()
            self.cnt = 0
            self.round += 1
            self.total = 0
        # self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        self.server_socket.close()

    def pull(self):
        while self.cnt < self.n_clients:
            client_socket, address = self.server_socket.accept()
            self.clients_socket.append(client_socket)

            client_para = b''
            tmp = client_socket.recv(1024)
            while tmp:
                if tmp.endswith(b'stop!'):
                    client_para += tmp[:-5]
                    break
                client_para += tmp
                tmp = client_socket.recv(1024)
            decode = pickle.loads(client_para)

            self.total += decode[0]
            self.para_cache.append(decode)

            self.cnt += 1
            print(f"SERVER@{self.ip}:{self.port} INFO: accept client@{address[0]}:{address[1]} parameters")

    def aggregate(self):
        clear_parameter(self.model)
        
        for key in self.model.state_dict():
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                # print(key)
                continue
            dtype = self.para_cache[0][1][key].dtype
            for idx in range(self.n_clients):
                self.model.state_dict()[key] += \
                    (self.para_cache[idx][0] / self.total) * self.para_cache[idx][1][key].to(dtype)
                
        acc1, acc5 = self.validate()
        print(f"SERVER@{self.ip}:{self.port} INFO: "
              f"Global Epoch[{self.round}|{self.global_epoch}]"
              f"Top-1 Accuracy: {acc1} "
              f"Top-5 Accuracy: {acc5}")
        # 以下为花费时间，不重要
        if(self.round==self.global_epoch):
            self.now_time=time.time()
            print(f"time: {time.asctime(time.localtime(self.now_time))}")
            print(seconds_to_hms(self.now_time-self.start_time))
        # 写入文件
        with open("D:\FL\Federated_Learning\Test.txt", 'a') as f:
            self.now_time=time.time()
            print(f"SERVER@{self.ip}:{self.port} INFO: "
              f"Global Epoch[{self.round}|{self.global_epoch}]"
              f"Top-1 Accuracy: {acc1} "
              f"Top-5 Accuracy: {acc5}"
              f" time: {time.asctime(time.localtime(self.now_time))}",file=f)
            print(seconds_to_hms(self.now_time-self.start_time),file=f)

    def push(self):
        for client_socket in self.clients_socket:
            client_socket.sendall(pickle.dumps(self.model.state_dict()))
            client_socket.sendall(b'stop!')
            client_socket.close()

    def validate(self):
        total = 0
        correct1 = 0
        correct2 = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.data):
                total += len(x)
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                predict = output.argmax(dim=1)
                correct1 += torch.eq(predict, y).sum().float().item()
                y_resize = y.view(-1, 1)
                _, predict = output.topk(5)
                correct2 += torch.eq(predict, y_resize).sum().float().item()
        return correct1 / total, correct2 / total
