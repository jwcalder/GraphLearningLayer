import torch
import torch.nn as nn
import torch.nn.functional as F

class customCNN(nn.Module):
    def __init__(self):
        super(customCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 10)
        # self.fc3 = nn.Linear(1024, 256)
        # self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 256 * 7 * 7)

        feat = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(feat), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        # x = F.relu(self.fc4(x))

        x = F.softmax(x, dim=1)
        return x, F.normalize(feat, dim=1)