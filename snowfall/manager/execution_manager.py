import torch


class ExecutionManager:
    def __init__(self, use_gpu=False, gpu_index=0, log_state=False):
        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU not available")
        self.use_gpu = use_gpu
        self.log_state = log_state
        self.cpu_d = torch.device("cpu")
        self.device = torch.device("cuda:%d" % gpu_index) if use_gpu else self.cpu()
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
