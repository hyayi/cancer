import torch.nn as nn 

class BaseClassify(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.classifier = nn.Sequential(
           nn.LeakyReLU(),
           nn.Linear(in_features=input_size, out_features=512),
           nn.LeakyReLU(),
           nn.Linear(in_features=512, out_features=256),
           nn.LeakyReLU(),
           nn.Linear(in_features=256, out_features=128),
           nn.LeakyReLU(),
           nn.Linear(in_features=128, out_features=64),
           nn.LeakyReLU(),
           nn.Linear(in_features=64, out_features=num_classes)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x