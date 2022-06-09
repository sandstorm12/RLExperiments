import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs,
            hidden_size=256, learning_rate=1e-3) -> None:
        super().__init__()

        self._initialize_parameters(num_inputs, num_outputs, hidden_size)

        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self.parameters(), learning_rate)

    def _initialize_parameters(self, num_inputs, num_outputs,
            hidden_size=256):
        self._fc1 = nn.Linear(num_inputs, hidden_size)
        self._fc2 = nn.Linear(hidden_size, hidden_size)
        self._fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)

        return x

    def train(self, outputs, labels):
        loss = self._criterion(outputs, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
