import pickle
import socket
import time
import torch
from torch.utils.data import DataLoader

from . import BaseClient

class ScaffoldClient(BaseClient):
    def __init__(
        self,
        ip: str,
        port: int,
        server_ip: str,
        server_port: int,
        model: str,
        data: DataLoader,
        sample_num: int,
        n_classes: int,
        global_epoch: int,
        local_epoch: int,
        optimizer: str,
        lr: float,
        device: str,
        criterion=torch.nn.CrossEntropyLoss()
    ):
        super(ScaffoldClient, self).__init__(
            ip,
            port,
            server_ip,
            server_port,
            model,
            data,
            sample_num,
            n_classes,
            global_epoch,
            local_epoch,
            optimizer,
            lr,
            device,
            criterion)
        self.local_control = None  # 新增：本地控制变量ci
        self.global_control = None   # 新增：全局控制变量c的副本
        self.first_data = None
        self.K= 0

    def first_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))
        # 接收服务器的新模型和全局控制变量        
        new_data = self.client_recv(client_socket)
        # 保留本地BN运行时统计量，仅更新其他参数
        local_state_dict = self.model.state_dict()
        for key in new_data['model']:
            if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
                local_state_dict[key] = new_data['model'][key]
        self.model.load_state_dict(local_state_dict)  # 关键修改
        self.first_data = new_data
        # 初始化本地控制变量ci为服务器发送的全局控制变量c
        self.global_control = {k: v.clone() for k, v in new_data['control'].items()}  
        self.local_control = {k: v.clone() for k, v in new_data['control'].items()} 
        # self.local_control = {k: 0 for k, v in data['control'].items()} #令ci为0
        client_socket.close()
        
    def train(self):
        loss_avg = 0
        cnt = 0
        for x, y in self.data:
            cnt += 1
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            
            # 计算梯度并应用SCAFFOLD修正：g_i + (c - ci)
            loss.backward()
            # with torch.no_grad():
            #     for name, param in self.model.named_parameters():
            #         c = self.global_control[name]
            #         ci = self.local_control[name]
            #         param.data += (ci - c)  # 修正项
            #         self.K += 1
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.grad += (self.global_control[name] - self.local_control[name])
                    self.K += 1
            self.optimizer.step()
            loss_avg += loss.item()
        return loss_avg / cnt

    def push_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))
        
        # 计算控制变量差值（Option II）
        control_plus = {}
        with torch.no_grad():
            for key in self.local_control:
                # Δc_i = (x - y_i)/(K*η_l) - (c - c_i)
                delta_model = (self.first_data['model'][key] - self.model.state_dict()[key])
                control_plus[key] = -delta_model / (self.K * self.lr) + self.local_control[key] - self.global_control[key]
        self.K=0
        data = {
            'sample_num': self.sample_num,
            'model': self.model.state_dict(), #yi
            'control_delta': control_plus #ci_plus
        }
        for key in self.local_control:
            self.local_control[key] = control_plus[key]
        
        client_socket.sendall(pickle.dumps(data))
        client_socket.sendall(b'stop!')
        
        # 接收服务器的新模型和全局控制变量        
        new_data = self.client_recv(client_socket)
        # 保留本地BN运行时统计量，仅更新其他参数
        local_state_dict = self.model.state_dict()
        for key in new_data['model']:
            if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
                local_state_dict[key] = new_data['model'][key]
        self.model.load_state_dict(local_state_dict)  # 关键修改
        self.global_control = new_data['control']
        client_socket.close()
