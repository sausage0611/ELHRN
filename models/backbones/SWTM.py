import torch
import torch.nn as nn


class SWTM(nn.Module):
    def __init__(self):
        super(SWTM, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 4)
        self.relu = nn.ReLU()

    def forward(self, way, shot):
        x = torch.tensor([[way, shot]], dtype=torch.float32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


