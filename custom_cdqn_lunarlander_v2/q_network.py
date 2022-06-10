import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchsummary import summary


class QNetwork(nn.Module):
    def __init__(self, num_outputs, learning_rate=1e-3) -> None:
        super().__init__()

        self._initialize_parameters(num_outputs)

        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self.parameters(), learning_rate)

    def _initialize_parameters(self, num_outputs):
        self.conv1 = nn.Conv3d(1, 32, 3)
        self.conv2 = nn.Conv3d(32, 32, (1, 3, 3))
        self.conv3 = nn.Conv3d(32, 64, (1, 3, 3))
        self.conv4 = nn.Conv3d(64, 64, (1, 3, 3))
        self.conv5 = nn.Conv3d(64, 64, (1, 3, 3))
        self.pool = nn.MaxPool3d((1, 2, 2))
        self._fc1 = nn.Linear(1600, 512)
        self._fc2 = nn.Linear(512, 128)
        self._fc3 = nn.Linear(128, num_outputs)

        summary(self, (1, 3, 224, 224))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)

        return x

    def train_step(self, outputs, labels):
        loss = self._criterion(outputs, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
