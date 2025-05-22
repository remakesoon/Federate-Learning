import os
import random

import numpy as np
import torch


def clear_parameter(model: torch.nn.Module):
    for key in model.state_dict():
        if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
            torch.nn.init.zeros_(model.state_dict()[key])


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def seconds_to_hms(seconds):
    #时间函数，秒->时分秒
    total = int(round(seconds))  # 四舍五入取整（或使用 int() 直接截断）
    hours, remaining = divmod(total, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"