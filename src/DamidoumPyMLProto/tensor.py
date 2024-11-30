from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from DamidoumPyMLProto.utils import is_valid_array


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

    @property
    def data(self) -> np.ndarray:
        """Getter for data attribute (np.ndarray) of the tensor

        Returns:
            np.ndarray: data attribute of the tensor
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Setter for data attribute (np.ndarray) of the tensor
        Check the validity of the input data (type, size and dtype)

        Args:
            value (np.ndarray): data to be set as the data attribute of the tensor
        """
        is_valid_array(value)
        self._data = value

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(
            data=Add().forward([self, other]),
            requires_grad=self.requires_grad or other.requires_grad,
            primitive=Add(),
            parent_nodes=[self, other],
        )

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor(
            data=Mul().forward([self, other]),
            requires_grad=self.requires_grad or other.requires_grad,
            primitive=Mul(),
            parent_nodes=[self, other],
        )

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, primitive={self.primitive})"


class Primitive(ABC):
    @staticmethod
    @abstractmethod
    def forward(inputs: list[Tensor]):
        """Perform the forward pass of the primitive

        Args:
            inputs (list[Tensor]): List of input tensors
        """
        pass

    @staticmethod
    @abstractmethod
    def backward(incomming_grad: np.ndarray, inputs: list[Tensor]):
        """Perform the backward pass

        Args:
            incomming_grad (np.ndarray): previous gradient
            inputs (list[Tensor]): List of input tensors
        """
        pass

    @classmethod
    def __repr__(cls):
        return f"<{cls.__name__}>"


class Add(Primitive):
    @staticmethod
    def forward(inputs: list[Tensor]):
        if len(inputs) != 2:
            raise ValueError("Add primitive requires exactly two inputs")
        if inputs[0].data.shape != inputs[1].data.shape:
            raise ValueError(
                "Shapes of input tensors must match"
            )  # TODO: Implement broadcasting
        return inputs[0].data + inputs[1].data

    @staticmethod
    def backward(incomming_grad, inputs: list[Tensor]):
        if len(inputs) != 2:
            raise ValueError("Add primitive requires exactly two inputs")
        if inputs[0].data.shape != inputs[1].data.shape:
            raise ValueError("Shapes of input tensors must match")
        return incomming_grad, incomming_grad


class Mul(Primitive):
    @staticmethod
    def forward(inputs: list[Tensor]):
        if len(inputs) != 2:
            raise ValueError("Mul primitive requires exactly two inputs")
        if inputs[0].data.shape[1] != inputs[1].data.shape[0]:
            raise ValueError("Shapes of input tensors must match")
        return inputs[0].data @ inputs[1].data

    @staticmethod
    def backward(incomming_grad, inputs: list[Tensor]):
        if len(input) != 2:
            raise ValueError("Mul primitive requires exactly two inputs")
        if inputs[0].data.shape[1] != inputs[1].data.shape[0]:
            raise ValueError("Shapes of input tensors must match")
        return incomming_grad @ inputs[1].data.T, inputs[0].data.T @ incomming_grad
