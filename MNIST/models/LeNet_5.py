import torch
import torch.nn as nn
from functions import *
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def preprocessing(x, w, padding):
    inp_unf = torch.nn.functional.unfold(x, (w.shape[2], w.shape[3]), padding = padding)
    A = inp_unf.transpose(1, 2)
    B = w.view(w.size(0), -1).t()
    return A,B

def postprocessing(x, exp_x, bias):
    x = x.transpose(1, 2)
    x = x.view(exp_x.shape)
    
    # bias
    bias = torch.broadcast_to(bias,(x.shape[0],x.shape[1])).reshape(x.shape[0],x.shape[1],1,1)
    x += bias
    return x

class LeNet_5(nn.Module):
    """This class defines a standard DNN model based on LeNet_5"""

    def __init__(self):
        """Initialization"""

        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU(inplace=True)
        return

    def forward(self, x):
        """Forward propagation procedure"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.reshape(-1, 256)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        return x
