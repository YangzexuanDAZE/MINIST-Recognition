# Firstly, let's download MNIST
import torch
import math
from pathlib import Path
import requests
import torch.nn.functional as F
import pickle
import gzip
import numpy as np

# Download Data
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)


# Then we load all training and cv set into 
# "x_train, y_train, x_valid, y_valid"
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape # n is number, c is hidden unit #


# Then we load x_train/y_train into ds & dl
# from torch.utils.data import TensorDataset, DataLoader

# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size= 1)
#   Here we choose the batch size to be 1, manually


# Hyperpara Here
lr = 0.5
bs = 64
epochs = 10

# Build Forward Pass using nn.Module
from torch import nn
class Mnist_Logistch(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10)) / math.sqrt(784)
        # Above is Xavier Initialization
        self.bias = nn.Parameter(torch.zeros(10))
    
    def forward(self, xb):
        return xb @ self.weights + self.bias

def init_weight():
    weights = torch.randn(784, 10) / math.sqrt(784) # Xavier init.
    weights.require_grad_()
    bias = torch.zeros(10, requires_grad= True)
    return weights, bias

# Evaluation Function Here
def accracy(y, yb):
    preds = torch.argmax(y, dim=1)
    return (preds == yb).float().mean()

loss_func = F.cross_entropy

xb = x_train[0:bs]
yb = y_train[0:bs]
model = Mnist_Logistch()
loss = loss_func(model(xb), yb)
print(f"Loss Function Calculated without training:{loss}" )
# opt = torch.optim.SGD(model.parameters(), lr= lr)
# for epoch in range(epochs):
#     for xb, yb in train_dl:
#         pred = model(xb)
#         loss = loss_func(pred, yb)
#         print(loss)

#         loss.backward()
#         opt.step()
#         opt.zero_grad()
