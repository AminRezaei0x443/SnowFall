class ExecutionEvents:
    state = {}

    def __init__(self):
        pass

    def set_model(self, model, optimizer):
        self.state["model"] = model
        self.state["optimizer"] = optimizer

    def on_epoch_start(self, epoch, **kwargs):
        if "epoch_start" in self.state:
            self.state["epoch_start"](epoch, **kwargs)
        pass

    def on_epoch_end(self, epoch, losses, avg_loss, **kwargs):
        if "epoch_end" in self.state:
            self.state["epoch_end"](epoch, losses, avg_loss, **kwargs)
        pass

    def on_batch_processed(self, epoch, batch, loss, **kwargs):
        if "batch_processed" in self.state:
            self.state["batch_processed"](epoch, batch, loss, **kwargs)
        pass

    def on_validated(self, epoch, loss, **kwargs):
        if "validated" in self.state:
            self.state["validated"](epoch, loss, **kwargs)
        pass

    def attach_epoch_start(self, func):
        self.state["epoch_start"] = func
        pass

    def attach_epoch_end(self, func):
        self.state["epoch_end"] = func
        pass

    def attach_batch_processed(self, func):
        self.state["batch_processed"] = func
        pass

    def attach_validated(self, func):
        self.state["validated"] = func
        pass
