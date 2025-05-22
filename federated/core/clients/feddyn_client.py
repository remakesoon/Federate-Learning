import socket
import pickle
import struct

import torch
from torch.utils.data import DataLoader

from . import BaseClient


class FedDynClient(BaseClient):
    def __init__(
            self,
            ip: str, # IP地址
            port: int, # 端口
            server_ip: str,
            server_port: int,
            model: str, # 模型名称
            data: DataLoader, # 数据加载器
            sample_num: int, # 样本数量
            n_classes: int, # 类别数量
            global_epoch: int, # 全局迭代轮次
            local_epoch: int, # 局部迭代轮次
            optimizer: str, #优化器类型
            lr: float, # 学习率
            device: str, # 设备类型
            criterion=torch.nn.CrossEntropyLoss(), # 损失函数 # 初始版本
            # criterion=ConsistencyCrossEntropyLoss() # 新版本
            alpha: float=0.005
    ):
        super(FedDynClient, self).__init__(
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
        #新改动start
        self.alpha = alpha  # 正则化强度
        self.h = None  # 本地梯度状态
        self.global_model = None  # 服务器全局模型副本
        #新改动end
        
        self.local_control = None  # 新增：本地控制变量ci
        self.global_control = None   # 新增：全局控制变量c的副本
        self.first_data = None
        self.K= 0

    def first_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))
        # 接收全局模型和h状态
        new_data = self.client_recv(client_socket)
        # 保留本地BN运行时统计量，仅更新其他参数
        local_state_dict = self.model.state_dict()
        for key in new_data['model']:
            if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
                local_state_dict[key] = new_data['model'][key]
        self.global_model = local_state_dict
        self.model.load_state_dict(self.global_model)
        self.h = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        self.first_data = new_data
        # 初始化本地控制变量ci为服务器发送的全局控制变量c
        self.global_control = {k: v.clone() for k, v in new_data['control'].items()}  
        self.local_control = {k: torch.zeros_like(v) for k, v in new_data['control'].items()} 
        # self.local_control = {k: 0 for k, v in data['control'].items()} #令ci为0
        
        client_socket.close()

    def train(self):
        loss_avg = 0
        cnt = 0
        theta_global = {k: v.clone() for k, v in self.global_model.items()}
        
        # 计算原始梯度
        original_grad = {}
        for x, y in self.data:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    original_grad[name] = param.grad.clone() / len(self.data)
            break  # 仅需一个batch计算近似梯度

        # FedDyn正则化训练
        for x, y in self.data:
            cnt += 1
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            
            # 原始损失项
            loss = self.criterion(output, y)
            
            # FedDyn正则项
            reg_loss = 0
            for name, param in self.model.named_parameters():
                reg_loss += torch.sum(-self.h[name].to(self.device) * param)
                reg_loss += (self.alpha / 2) * torch.norm(param - theta_global[name].to(self.device))**2               
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    c = self.global_control[name]
                    ci = self.local_control[name]
                    param.data += (ci - c)  # 修正项
                    self.K += 1
                    
            total_loss = loss + reg_loss
            total_loss.backward()
            loss_avg += total_loss.item()
            self.optimizer.step()
        
        with torch.no_grad():
            new_h = {}
            for name, param in self.model.named_parameters():
                new_h[name] = original_grad[name] - self.alpha * (param.data.to(self.device) - theta_global[name])
            self.h = new_h
            
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
                control_plus[key] = delta_model / (self.K * self.lr) + self.local_control[key] - self.global_control[key]
        self.K=0
        # 发送样本数、模型参数和梯度状态
        data = {
            'sample_num': self.sample_num,
            'model': self.model.state_dict(),
            'control_delta': control_plus #ci_plus
        }
        client_socket.sendall(pickle.dumps(data))
        client_socket.sendall(b'stop!')
        
        # 接收新的全局状态
        #接受
        new_data = self.client_recv(client_socket)
        # 保留本地BN运行时统计量，仅更新其他参数
        local_state_dict = self.model.state_dict()
        for key in new_data['model']:
            if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
                local_state_dict[key] = new_data['model'][key]
        self.global_model = local_state_dict
        self.model.load_state_dict(self.global_model)
        
        self.global_control = {k: v.clone() for k, v in new_data['control'].items()} 
        
        client_socket.close()