from DamidoumPyMLProto import Tensor
import numpy as np

if __name__ == "__main__":
    x = Tensor(data=np.array([[2.0, 1.0], [2.0, 3.0]]), requires_grad=True)
    w = Tensor(data=np.array([[3.0], [1.0]]), requires_grad=True)
    b = Tensor(data=np.array([[4.0], [9.0]]), requires_grad=True)
    l = x * w + b
    print(l)
