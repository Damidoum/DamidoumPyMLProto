from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Tensor:
    """
    A tensor is a multi-dimensional array part of a computatnioal graph.
    """

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        primitive: Primitive | None = None,
        parent_nodes: list[Tensor] | None = None,
    ):
        self.data = data
        self.requires_grad = requires_grad
        self.primitive = primitive
        self.parent_nodes = parent_nodes
        self.grad = np.zeros_like(data)

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(
            data=Add().forward(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            primitive=Add(),
            parent_nodes=[self, other],
        )

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor(
            data=Mul().forward(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
            primitive=Mul(),
            parent_nodes=[self, other],
        )

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, primitive={self.primitive})"


class Primitive(ABC):
    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def backward(self, incomming_grad, *inputs):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class Add(Primitive):
    def forward(self, *inputs):
        pass

    def backward(self, incomming_grad, *inputs):
        pass


class Mul(Primitive):
    def forward(self, *inputs):
        pass

    def backward(self, incomming_grad, *inputs):
        pass
