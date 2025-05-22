import socket
import pickle
import struct
import time
import torch
from torch.utils.data import DataLoader

from . import BaseClient


class FedProxCtrlClient(BaseClient):
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
            criterion=torch.nn.CrossEntropyLoss(),
            MU: float = 0.1, #新参数
            Beta: float = 0.15
    ):
        super(FedProxCtrlClient, self).__init__(
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
        self.model_parameter = None
        self.pre_parameter = None
        self.MU = MU #新参数μ
        self.Beta = Beta
        self.Gamma = self.MU+self.Beta
    # run未修改
    def train(self):
        loss_avg = 0
        cnt = 0
        for x, y in self.data:
            cnt += 1
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            #新行为开始
            proximal_term = 0.0
            control = 0.0
            #dyn_term = 0.0
            for w, w_t, w_p in zip(self.model_parameter, self.model.parameters(),self.pre_parameter):
                proximal_term +=  (w - w_t).norm(2) 
                control += (w - w_p).norm(2)
                #dyn_term += (w - (self.MU*w_t+self.Beta*w_p)/self.Gamma).norm(2)
            # loss = self.criterion(output, y) + self.Gamma*dyn_term/2
            loss = self.criterion(output, y) + self.MU*proximal_term/2 + self.Beta*control/2
            #新的目标函数
            #新行为结束
            loss.backward()
            loss_avg += loss.item()
            self.optimizer.step()
        return loss_avg / cnt

    def push_pull(self):
        #发送
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))
        client_socket.sendall(pickle.dumps([self.sample_num, self.model.state_dict()]))
        client_socket.sendall(b'stop!')
        
        self.pre_parameter = [param.data.clone() for param in self.model.parameters()]
        
        #接受
        global_state_dict = self.client_recv(client_socket)
        # 保留本地BN运行时统计量，仅更新其他参数
        local_state_dict = self.model.state_dict()
        for key in global_state_dict:
            if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
                local_state_dict[key] = global_state_dict[key]
        self.model.load_state_dict(local_state_dict)  # 关键修改
        # self.model_parameter = self.model.parameters() #新行为
        self.model_parameter = [param.data.clone() for param in self.model.parameters()]

        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client_socket.close()

    def first_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))

        global_state_dict = self.client_recv(client_socket)
        # 保留本地BN运行时统计量，仅更新其他参数
        local_state_dict = self.model.state_dict()
        for key in global_state_dict:
            if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
                local_state_dict[key] = global_state_dict[key]
        self.model.load_state_dict(local_state_dict)  # 关键修改
        # self.model_parameter = self.model.parameters() #新行为
        self.model_parameter = [param.data.clone() for param in self.model.parameters()]
        self.pre_parameter = [param.data.clone() for param in self.model.parameters()]
        
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client_socket.close()
