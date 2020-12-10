import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 3)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        # # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(18, 8) #input layer of 18 to 8
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

        # self.fc2 = nn.Linear(120, 84) #size of output? 2


    def forward(self, x):
        # Max pooling over a (2, 2) window // pull x from numpy array

        #...

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # sigmoid function: x = self.sigmoid(x)
        x = self.sigmoid(x)
        return x

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))



if __name__ == '__main__':
    x = torch.rand(5, 3)
    print(x)
    # print(sigmoid(0.5))

    net = Net()
    print(net)

    # have config file with pretty prints, config files etc.
    # Pass in net and data to training function.