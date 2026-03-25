import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engineering.value import Value


class Neuron:
    def __init__(self, n):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(n)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        total_products = sum((wi * xi for wi, xi in zip(self.w, x)), Value(0.0))
        logit = total_products + self.b
        return logit.tanh()
    
    def parameters(self):
        return self.w + [self.b]


class Linear:
    def __init__(self, in_features, out_features):
        self.neurons = [Neuron(in_features) for _ in range(out_features)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    

class MLP:
    def __init__(self, in_channels, hidden_channels):
        sizes = [in_channels] + hidden_channels
        self.layers = [Linear(sizes[i], sizes[i+1]) for i in range(len(hidden_channels))] 

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]