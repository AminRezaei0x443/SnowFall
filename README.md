## SnowFall
Easy to use ML Framework based on PyTorch. Train and validate your models easily without any boilerplate codes.

### How to Use
1. Install the package
```shell
pip install snowfall-ml
```
2. Import the modules
```python
from snowfall.manager.execution_manager import ExecutionManager
from snowfall.nn import SnowModule, Network
import torch.nn as nn
```
3. Create networks and train them (You can easily use operators syntax to fastly create and train your SnowModule or just use the old-fashion methods)
```python
from snowfall.generalization import DisturbLabel, EarlyStopping, L2Regularize

net = Network()
net += nn.Linear(4, 12)
net += nn.Linear(12, 36)
net += nn.Linear(36, 3)

manager = ExecutionManager(use_gpu=False)

def disturb(x, y):
    return x, disturber(y)

def accuracy(pred, gt):
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(gt.view_as(pred)).sum()
    return correct.float() / pred.shape[0]

c = SnowModule(manager)
# Add The Modules
c += net
# Multiply the Optimizer
c *= "adam"
# Reduce the loss/cost
c -= L2Regularize(nn.CrossEntropyLoss(), 0.001)
# Calculate the metrics
c %= ("acc", accuracy)
# Pipe the preprocessors
c |= (disturb, False)
# Power up the event listeners
c **= EarlyStopping(patience=10)
# Learn from the dataset
c.learn(trainSet, epochs=24, train_batch=4, val_batch=4)
```

4. You can see the "examples" directory to see working examples.
### Todos
This project is still in beta and is a WIP. Thus, It will go through major API changes and upgrades.

- [ ] Documentation
- [ ] Better API
- [ ] Mixed Precision and Quantization Aware Training
- [ ] Visualizations
- [ ] Add Non Neural Network Models
- [ ] Multi-GPU and Distributed Training
- [ ] Model Serving
- [ ] etc...

### Contributions
Contributions are welcome, You can contact me at [AminRezaei0x443@gmail.com](mailto:AminRezaei0x443@gmail.com) if you want to have a talk about the project and ideas, ... .