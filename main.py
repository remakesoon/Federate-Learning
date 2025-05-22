import argparse
import torch
from federated.core.configs import Config
from federated.core.utils import seed_it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./examples/FedProxCtrl/config_mp_cifar10.yml", type=str, help='config file')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--arch', default="mp", type=str)
    arg = parser.parse_args()
    seed_it(arg.seed)
    config = Config(arg.config)
    with open("D:\FL\Federated_Learning\Test.txt", 'a') as f:
        print(f"\nseed: {arg.seed}",file=f)
    # if torch.cuda.is_available():
    #     print("GPU is available!")
    # else:
    #     print("GPU is not available.")

    if arg.arch == "mp":
        config.run_mp()
    else:
        config.run_distributed()


if __name__ == '__main__':
    main()
