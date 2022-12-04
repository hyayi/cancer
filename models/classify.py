import torch.nn as nn 

class BaseClassify(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x