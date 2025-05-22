from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from . import Data


class Cifar10(Data):
    def __init__(self,
                 n_clients: int,
                 batch_size: int,
                 path: str,
                 alpha: float = 100,
                 plot_path="D:\FL\Federated_Learning\data",
                 flag=True):
        super().__init__(plot_path)
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        self.train_set = CIFAR10(root=path, train=True, transform=transform, download=True)
        self.validate_set = CIFAR10(root=path, train=False, transform=transform)
        self.trainLoader, self.client_nums, self.total = \
            self.train_loader(alpha, n_clients, batch_size, flag)
        self.validationLoader = self.validate_loader(batch_size * n_clients)

    def __str__(self):
        return "CIFAR10"
