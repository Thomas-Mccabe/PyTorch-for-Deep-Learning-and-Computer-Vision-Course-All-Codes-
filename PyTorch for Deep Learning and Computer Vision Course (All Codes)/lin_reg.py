import torch
from torch import optim
from torch.autograd import grad
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Linear
from icecream import ic
from torch.nn.modules import module

# Define Linear Regression class
# nn.Modulse is a base class
class LR(nn.Module):
    def __init__(self, input_size: int =1, output_size: int = 1) -> nn.Module:
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred

def tute17():
    def get_params():
        return w[0][0].item(), b[0].item()

    def plot_fit(title):
        w1, b1 = get_params()
        x1 = np.array([-30, 30])
        y1 = w1*x1+b1

        plt.title(title)
        plt.plot(X.numpy(), y.numpy(), 'o')
        plt.plot(x1, y1, 'r')
        plt.show()

    # Make a noisy input datset
    X = torch.randn(100, 1)*10
    y = X + 3*torch.randn(100, 1)
    plt.ylabel('y')
    plt.xlabel('x')

    torch.manual_seed(1)
    model = LR(1, 1)
    ic(model)
    [w, b] = model.parameters()
    ic(w, b)

    plot_fit('Initial Model')



def tute14():
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

def tute16():
    model = LR(1, 1)
    ic(list(model.parameters()))

    x = torch.tensor([[1.0], [2.0]])
    pred = model.forward(x)
    ic(pred)

# Loss Functions
# distance error from y_hat to y
# 
def tute21():
    torch.manual_seed(1)
    X = torch.randn(100, 1)*10
    y = X + 3*torch.randn(100, 1)

    model = LR(1, 1)
    criteon = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01) # stocastic gradient descent. Minimises loss per each sample

    epochs = 100 # too few -> underfitting, too many -> overfitting
    losses = []
    for i in range(epochs):
        y_pred = model.forward(X)
        loss = criteon(y_pred, y)
        print(f"epoch {i}, loss: {loss.item()}")

        losses.append(loss)
        optimiser.zero_grad()
        loss.backward() # This needs to always be done before step
        optimiser.step()

    plt.plot(range(epochs), losses)
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.show()
#run = tute17()
run = tute21()