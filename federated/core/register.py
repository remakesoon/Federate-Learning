from ..core.clients import *
from ..core.server import *
from ..datasets import *
from ..models import *

all_optim = {"SGD": torch.optim.SGD}
all_arch = {"SimpleCNN": SimpleCNN, "VGG11": VGG11, "VGG16": VGG16, "ResNet18": Resnet18, "SimpleFC": SimpleFC , "SimpleDNN": SimpleDNN}
all_data = {"MNIST": Mnist, "CIFAR10": Cifar10}
all_server = {"FedAVG": BaseServer, "FedProx": BaseServer, "FedProxCtrl": BaseServer, "Scaffold": ScaffoldServer ,"FedDyn": FedDynServer,"FedDynNew": FedDynServerNew,"ScaffoldNew": ScaffoldServerNew}
all_client = {"FedAVG": BaseClient, "FedProx": FedProxClient, "FedProxCtrl": FedProxCtrlClient, "Scaffold": ScaffoldClient,"FedDyn": FedDynClient,"FedDynNew": FedDynClientNew,"ScaffoldNew": ScaffoldClientNew}