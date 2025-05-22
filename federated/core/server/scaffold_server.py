import pickle
import socket
import struct
import time
import torch
from torch.utils.data import DataLoader

from . import BaseServer

from ..utils import clear_parameter,seconds_to_hms

class ScaffoldServer(BaseServer):
    def __init__(
        self,
        ip: str,
        port: int,
        global_epoch: int,
        n_clients: int,
        model: str,
        data: DataLoader,
        n_classes: int,
        device: str
    ):
        super(ScaffoldServer,self).__init__(
            ip,
            port,
            global_epoch,
            n_clients,
            model,
            data,
            n_classes,
            device,
        )
        self.control_variates_cache = []  # 新增：存储客户端控制变量更新
        self.global_control = None        # 新增：全局控制变量c
        self.client_sample_nums = []  # 新增：存储各客户端的sample_num

    def first_push(self):
        while self.cnt < self.n_clients:
            client_socket, address = self.server_socket.accept()
            self.clients_socket.append(client_socket)
            self.cnt += 1
        # 发送初始模型和全局控制变量
        # 初始化全局控制变量为与模型参数同形状的零张量
        self.global_control = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        for client_socket in self.clients_socket:
            data = {
                'model': self.model.state_dict(),
                'control': self.global_control
            }
            client_socket.sendall(pickle.dumps(data))
            client_socket.sendall(b'stop!')
            client_socket.close()
            
        self.cnt = 0
        self.clients_socket.clear()

    def pull(self):
        while self.cnt < self.n_clients:
            client_socket, address = self.server_socket.accept()
            self.clients_socket.append(client_socket)

            client_para = b''
            tmp = client_socket.recv(4096)
            while tmp:
                if tmp.endswith(b'stop!'):
                    client_para += tmp[:-5]
                    break
                client_para += tmp
                tmp = client_socket.recv(4096)
            decode = pickle.loads(client_para)

            self.total += decode['sample_num']
            self.para_cache.append(decode)

            self.cnt += 1
            print(f"SERVER@{self.ip}:{self.port} INFO: accept client@{address[0]}:{address[1]} parameters")

    def aggregate(self):
        clear_parameter(self.model)
        # 聚合模型参数
        
        # i
        for key in self.model.state_dict():
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                # print(key)
                continue
            dtype = self.para_cache[0]['model'][key].dtype
            for idx in range(self.n_clients):
                self.model.state_dict()[key] += (self.para_cache[idx]['sample_num'] / self.total) * self.para_cache[idx]['model'][key].to(dtype)
        
        for key in self.global_control:
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                # print(key)
                continue
            delta = torch.zeros_like(self.global_control[key])
            for idx in range(self.n_clients):
                delta += self.para_cache[idx]['control_delta'][key]
            self.global_control[key] =self.global_control[key] + delta / self.n_clients
        # ii
        
            
        # 验证并打印结果
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
            data = {
                'model': self.model.state_dict(),
                'control': self.global_control
            }
            client_socket.sendall(pickle.dumps(data))
            client_socket.sendall(b'stop!')
            client_socket.close()
