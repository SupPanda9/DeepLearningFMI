from abc import ABC, abstractmethod
import torch


class Module(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


class Sigmoid(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))


class Tanh(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class ReLU(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(
            torch.tensor(0.0), x
        ) + self.negative_slope * torch.minimum(torch.tensor(0.0), x)


class Sequential(Module):
    def __init__(self, *modules: Module):
        self._modules = list(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self._modules:
            x = module(x)
        return x

    def append(self, module: Module):
        self._modules.append(module)

    def extend(self, sequential: "Sequential"):
        self._modules.extend(sequential._modules)

    def insert(self, index: int, module: Module):
        self._modules.insert(index, module)
