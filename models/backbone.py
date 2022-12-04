from torchvision.models import resnet50
import torch
import torch.nn as nn


class ResNet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential(*(list(self.model.children())[:-1]))
        self.output_size = self.model.fc.in_features
        del self.model
    
    def forward(self, x):
        patch_features = self.extractor(x) 
        return patch_features 
        

        