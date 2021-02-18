
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # # an affine operation: y = Wx + b
        self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(18, 12) # input layer of 18 to 8
        # self.linear2 = nn.Linear(32, 64)
        # self.linear3 = nn.Linear(64, 32)
        # self.linear4 = nn.Linear(32, 16)
        # self.linear5 = nn.Linear(16, 12)
        # self.linear6 = nn.Linear(12, 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 2)
        self.linear4 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        # x = self.linear5(x)
        # x = self.linear6(x)
        # x = self.linear7(x)
        # x = self.linear8(x)
        # x = self.linear9(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        # x = self.linear2_5(x)
        # # x = self.relu(x)
        # x = self.linear3(x)
        # x = self.relu(x)

        # sigmoid function: x = self.sigmoid(x)
        x = self.sigmoid(x)

        return x



if __name__ == '__main__':
    # Create the network and look at its text representation
    model = Net()
    print(model)


    # have config file with pretty prints, config files etc.
    # Pass in net and data to training function.