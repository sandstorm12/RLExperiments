import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, 256)
        # self.fc2 = nn.Linear(16, 32)
        # self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(256, num_outputs)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), 1e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def train(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
