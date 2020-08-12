from tqdm import tqdm
import torch.nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from data.lambda_loader import LambdaDataLoader
from events.execution import ExecutionEvents
from regularizations.disturb_label import DisturbLabel
from regularizations.early_stopping import EarlyStopping


class Executor:
    def __init__(self, model, manager):
        self.model = model.to(manager.current())
        self.manager = manager

    def configure(self, optimizer, loss="cross_entropy", preprocessor=None):
        self.optimizer = optimizer
        if isinstance(loss, str):
            if loss == "mse":
                self.criterion = F.mse_loss
            elif loss == "cross_entropy":
                self.criterion = F.cross_entropy
            else:
                raise Exception("Undefined loss func, use function or nn.Module instead!")
        elif isinstance(loss, nn.Module) or type(loss).__name__ == "function":
            self.criterion = loss
        else:
            raise Exception("loss parameter must be string, function or nn.Module!")

        if preprocessor is None:
            self.preproc = False
            pass
        elif type(preprocessor).__name__ == "function":
            self.preproc = True
            self.preprocessor = preprocessor
        else:
            raise Exception("Preprocessor type not recognized!")

    def train(self, train_dataset, epochs, val_split=0.15, train_batch=32, val_batch=32,
              event_listener=None, progress=True, metrics={},
              early_stopping=False, patience=5, disturb=False):
        total = len(train_dataset)
        val_count = int(val_split * total)
        self.train_set, self.val_set = random_split(train_dataset, (total - val_count, val_count))

        train_loader = DataLoader(self.train_set, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(self.val_set, batch_size=val_batch, shuffle=True)

        if self.preproc:
            train_loader = LambdaDataLoader(train_loader, self.preprocessor)
            val_loader = LambdaDataLoader(val_loader, self.preprocessor)
        event_listener = ExecutionEvents() if event_listener is None else event_listener
        event_listener.set_model(self.model, self.optimizer)
        if early_stopping:
            self.es = EarlyStopping(patience=patience)
        if disturb:
            self.disturb = DisturbLabel(alpha=20, C=10)
        for i in range(epochs):
            # Training an epoch
            self.model.train()
            losses = {}
            metrics_v = {}
            for metric in metrics:
                metrics_v[metric] = {}
            it = enumerate(train_loader)
            it = tqdm(it, total=len(train_loader), position=0) if progress else it
            if progress:
                it.set_description('Epoch %d' % i)
            event_listener.on_epoch_start(i)
            for batch_idx, (data, target) in it:
                data, target = data.to(self.manager.current()), target.to(self.manager.current())
                if disturb:
                    target = self.disturb(target).to(self.manager.current())

                def fwd():
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    losses[batch_idx] = loss.item()
                    for metric in metrics:
                        metrics_v[metric][batch_idx] = metrics[metric](output, target).item()
                    loss.backward(create_graph=True)
                    return loss, output

                self.optimizer.step(closure=fwd)
                event_listener.on_batch_processed(i, batch_idx, losses[batch_idx])
                if progress:
                    m_mean = {k: np.mean(list(metrics_v[k].values())) for k in metrics}
                    it.set_postfix(batch=batch_idx, loss=losses[batch_idx], mean_loss=np.mean(list(losses.values())),
                                   **m_mean)
            if progress:
                it.close()
            ls = list(losses.values())
            m_mean = {k: np.mean(list(metrics_v[k].values())) for k in metrics}
            m_all = {k + "_all": list(metrics_v[k].values()) for k in metrics}
            event_listener.on_epoch_end(i, ls, np.mean(ls), **m_mean, **m_all)
            # Validating an epoch
            self.model.eval()

            it = val_loader
            it = tqdm(it, total=len(val_loader), position=0) if progress else it
            if progress:
                it.set_description('Validating Epoch %d' % i)

            loss_val = []
            metrics_val = {}
            for metric in metrics:
                metrics_val[metric] = []
            with torch.no_grad():
                for data, target in it:
                    data, target = data.to(self.manager.current()), target.to(self.manager.current())
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss.item()
                    loss_val.append(loss)
                    for metric in metrics:
                        metrics_val[metric].append(metrics[metric](output, target).item())
                    if progress:
                        m_mean = {k: np.mean(metrics_val[k]) for k in metrics}
                        it.set_postfix(loss=loss, mean_loss=np.mean(loss_val),
                                       **m_mean)
            m_mean = {k: np.mean(metrics_val[k]) for k in metrics}
            m_all = {k + "_all": metrics_val[k] for k in metrics}
            event_listener.on_validated(i, np.mean(loss_val), **m_mean, **m_all)
            if early_stopping:
                if self.es.step(torch.tensor(np.mean(loss_val))):
                    break
            if progress:
                it.close()

    def eval(self, test_dataset, batch_size=32, progress=True, metrics={}):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        if self.preproc:
            test_loader = LambdaDataLoader(test_loader, self.preprocessor)

        model.eval()

        it = test_loader
        it = tqdm(it, total=len(test_loader), position=0) if progress else it
        if progress:
            it.set_description("Evaluating")

        loss_val = []
        metrics_val = {}
        for metric in metrics:
            metrics_val[metric] = []
        outs = []
        with torch.no_grad():
            for data, target in it:
                data, target = data.to(self.manager.current()), target.to(self.manager.current())
                output = self.model(data)
                outs.append(output)
                loss = self.criterion(output, target).item()
                loss_val.append(loss)
                for metric in metrics:
                    metrics_val[metric].append(metrics[metric](output, target).item())
                if progress:
                    m_mean = {k: np.mean(metrics_val[k]) for k in metrics}
                    it.set_postfix(loss=loss, mean_loss=np.mean(loss_val),
                                   **m_mean)
        m_mean = {k: np.mean(metrics_val[k]) for k in metrics}
        m_mean["loss"] = np.mean(loss_val)
        if progress:
            it.close()
        return torch.cat(outs), m_mean