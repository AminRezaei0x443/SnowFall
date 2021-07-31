class LambdaDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


class LambdasDataLoader:
    def __init__(self, dl, funcs=None):
        if funcs is None:
            funcs = []
        self.dl = dl
        self.funcs = funcs

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            x = b
            for f in self.funcs:
                x = f(*x)
            yield x
