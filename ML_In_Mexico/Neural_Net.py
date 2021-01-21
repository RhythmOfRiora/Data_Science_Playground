
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # # an affine operation: y = Wx + b
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(18, 8) # input layer of 18 to 8
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        # sigmoid function: x = self.sigmoid(x)
        x = self.sigmoid(x)

        return x



if __name__ == '__main__':
    # Create the network and look at its text representation
    model = Net()
    print(model)


    # have config file with pretty prints, config files etc.
    # Pass in net and data to training function.