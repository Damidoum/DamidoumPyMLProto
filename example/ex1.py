from DamidoumPyMLProto import Tensor

if __name__ == "__main__":
    x = Tensor(data=2.0, requires_grad=True)
    y = Tensor(data=3.0, requires_grad=True)
    z = x + y
    print(z)
