import torch.nn as nn
import torch
from data_load_utils import FENDataset
from torchvision import transforms

testing_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), 
                                                             (0.5, 0.5, 0.5))])


class Model(nn.Module):
    def __init__(self): 
        super().__init__() 
        self.conv1 = nn.Conv2d(16, 16, 3) # Input Channels, Number of Kernels, Kernel Size 
        self.pool = nn.MaxPool2d(3, 3) # Kernel Size, Stride 
        self.conv2 = nn.Conv2d(6, 16, 5) 
        
        self.fc1 = nn.Linear(16*5*5, 120) 
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10)
        self.sm = nn.LogSoftmax(dim =1) # this, at least, won't work

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) 
        x = self.sm(x)
        return x
