import torch


class GlobalManager:
    def __init__(self, use_gpu=False, log_state=False):
        self.use_gpu = use_gpu
        self.log_state = log_state
        self.cpu_d = torch.device("cpu")
        self.device = torch.device("cuda") if use_gpu else self.cpu
        pass

    def is_gpu(self):
        return self.use_gpu

    def current(self):
        return self.device

    def cpu(self):
        return self.cpu_d

    def log(self, msg):
        if self.log_state:
            print(msg)
