from snowfall.core import SnowObject


class ExecutionEvents(SnowObject):
    def __init__(self):
        super().__init__()

    def on_epoch_start(self, epoch, **kwargs):
        if "epoch_start" in self:
            self.epoch_start(epoch, **kwargs)
        pass

    def on_epoch_end(self, epoch, losses, avg_loss, **kwargs):
        if "epoch_end" in self:
            self.epoch_end(epoch, losses, avg_loss, **kwargs)
        pass

    def on_batch_processed(self, epoch, batch, loss, **kwargs):
        if "batch_processed" in self:
            self.batch_processed(epoch, batch, loss, **kwargs)
        pass

    def on_validated(self, epoch, loss, **kwargs):
        if "validated" in self:
            self.validated(epoch, loss, **kwargs)
        pass

    def attach_epoch_start(self, func):
        self.add_prop("epoch_start", func)
        pass

    def attach_epoch_end(self, func):
        self.add_prop("epoch_end", func)
        pass

    def attach_batch_processed(self, func):
        self.add_prop("batch_processed", func)
        pass

    def attach_validated(self, func):
        self.add_prop("validated", func)
        pass
