# Firstly, let's download MNIST
import torch
import math
from pathlib import Path
import requests
import torch.nn.functional as F
import pickle
import gzip
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
print("Data Loaded Successfully!\n")


# Then we load x_train/y_train into ds & dl
# from torch.utils.data import TensorDataset, DataLoader

# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size= 1)
#   Here we choose the batch size to be 1, manually

# Build Forward Pass using nn.Module
class Mnist_Logistch(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10)/ math.sqrt(784) ) 
        # Above is Xavier Initialization
        self.bias = nn.Parameter(torch.zeros(10))
    
    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistch()
# Hyperpara and Settings Here
lr = 0.4
bs = 64
epochs = 10
opt = torch.optim.SGD(model.parameters(), lr=lr)
print(f"HyperParameters set as: lr= {0.5}, bs= {64}, epochs= {epochs}, opt= {opt}, \n")

# Evaluation Function Here
def accracy(y, yb):
    preds = torch.argmax(y, dim=1)
    return (preds == yb).float().mean()

print(f"Accuracy without training: {accracy(model(x_valid), y_valid)}" )

loss_func = F.cross_entropy
# If you want to use fit function, define DS and DL firstly
# Define DS and DL Here
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size= bs)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size= bs)
print(f"Data loaded into DataLoader successfully! \n")

def loss_batch(model, loss_func, xb, yb, opt= None):
    y = model(xb)
    loss = loss_func(y, yb)

    if opt is not None:
        loss.backward(retain_graph= True)
        opt.step()
        opt.zero_grad()
    
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, train_dl, opt, valid_dl):
    for epoch in range(epochs):
        model.train()
        ls_sum = 0
        i_sum = 0
        for xb, yb in train_dl:
            ls, i = loss_batch(model, loss_func, xb, yb, opt= opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xv, yv) for xv, yv in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"Epoch: {epoch+1}, Losses in validation: {val_loss}")

# Finally, let's run our logistic Model
fit(epochs, model, loss_func, train_dl, opt, valid_dl)

print(f"Accuracy After Training: {accracy(model(x_valid), y_valid)} ")

# for epoch in range(epochs):
#     for xb, yb in train_dl:
#         pred = model(xb)
#         loss = loss_func(pred, yb)
#         print(loss)

#         loss.backward()
#         opt.step()
#         opt.zero_grad()
