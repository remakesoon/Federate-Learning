import pickle
import socket
import struct
import time

import torch
from torch.utils.data import DataLoader

from . import BaseServer

from ..utils import clear_parameter,seconds_to_hms

class FedDynServerNew(BaseServer):
    def __init__(
            self,
            ip: str,
            port: int,
            global_epoch: int, # 全局迭代轮次
            n_clients: int,
            model: str,
            data: DataLoader,
            n_classes: int,
            device: str,
            alpha: float = 0.03
    ):
        super(FedDynServerNew,self).__init__(
            ip,
            port,
            global_epoch,
            n_clients,
            model,
            data,
            n_classes,
            device,
        )
        #新内容start
        self.alpha = alpha
        self.h_global = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

    def first_push(self):
        while self.cnt < self.n_clients:
            client_socket, address = self.server_socket.accept()
            self.clients_socket.append(client_socket)
            self.cnt += 1
            
        # 初始化发送全局模型   
        for client_socket in self.clients_socket:
            data = {
                'model': self.model.state_dict(),
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
            tmp = client_socket.recv(1024)
            while tmp:
                if tmp.endswith(b'stop!'):
                    client_para += tmp[:-5]
                    break
                client_para += tmp
                tmp = client_socket.recv(1024)
            decode = pickle.loads(client_para)

            self.total += decode['sample_num']
            self.para_cache.append(decode)

            self.cnt += 1
            print(f"SERVER@{self.ip}:{self.port} INFO: accept client@{address[0]}:{address[1]} parameters")
            
    def aggregate(self):
        # 聚合模型参数
        clear_parameter(self.model)       
        # 聚合h状态
        for p in self.para_cache:
            for key in self.h_global:
                self.h_global[key] -= self.alpha*p['model'][key].to(self.device) / self.n_clients
            
        for key in self.h_global:    
            self.h_global[key] += self.alpha*self.model.state_dict()[key]/self.n_clients
        
        # 计算加权平均模型       
        for key in self.model.state_dict():
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                # print(key)
                continue
            dtype = self.para_cache[0]['model'][key].dtype
            for idx in range(self.n_clients):
                self.model.state_dict()[key] += (self.para_cache[idx]['sample_num'] / self.total) * self.para_cache[idx]['model'][key].to(dtype)
            self.model.state_dict()[key]-=self.h_global[key]
            
        # 验证精度
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
        data = {
            'model': self.model.state_dict(),
        }
        for client_socket in self.clients_socket:
            client_socket.sendall(pickle.dumps(data))
            client_socket.sendall(b'stop!')
            client_socket.close()