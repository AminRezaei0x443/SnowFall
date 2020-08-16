class SnowObject:
    data = {}
    reserved = []

    def __init__(self):
        self.data = {}
        self.reserved = []

    def add_prop(self, key, value):
        self.data[key] = value

    def get_prop(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item in self.data

    def __getattr__(self, item):
        if item in self.reserved:
            raise AttributeError("")
        return self.data[item]

    def reserve_prop(self, p):
        self.reserved.append(p)
