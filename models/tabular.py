
import torch.nn as nn

class TabularFeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size=512):
        super(TabularFeatureExtractor, self).__init__()
        self.output_size = output_size
        self.embedding = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=output_size)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        return x