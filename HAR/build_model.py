import torch.nn as nn
import torch.nn.functional as F


class HARModel(nn.Module):
    def __init__(self):
        super(HARModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(561, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        output = self.model(x)
        return output


class HARCNN(nn.Module):
    def __init__(self):
        super(HARCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (561 // 2 // 2), 128)  # Adjust based on input size after pooling
        self.fc2 = nn.Linear(128, 6)  # Number of classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor while keeping batch size consistent
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
