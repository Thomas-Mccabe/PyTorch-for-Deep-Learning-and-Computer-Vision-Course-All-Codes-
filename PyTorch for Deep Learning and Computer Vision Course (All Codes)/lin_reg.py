import torch
from torch.autograd import grad
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Linear
from icecream import ic

def tute1():
    X = torch.randn(100, 1)*10
    y = X + 3*torch.randn(100, 1)
    plt.plot(X.numpy(), y.numpy(), 'o')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    def forward(x, w ,b):
        y = w*x + b
        return y

    w = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)

    torch.manual_seed(1) # Fixes random seed to 1, canLibe a different seed

    # Linear model with 1 input and 1 output
    model = Linear(in_features=1, out_features=1)
    ic(model.bias)
    ic(model.weight)

    x = torch.tensor([[2.0], [3.3]])
    ic(model(x))

# Define Linear Regression class
# nn.Modulse is a base class
class LR(nn.Module):
    def __init__(self, input_size: int =1, output_size: int = 1) -> nn.Module:
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred

model = LR(1, 1)
ic(list(model.parameters()))

x = torch.tensor([[1.0], [2.0]])
pred = model.forward(x)
ic(pred)