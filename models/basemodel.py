import torchvision.models as models
import torch.nn as nn 
import torch 

class ImgFeatureExtractor(nn.Module):
    def __init__(self,model_name):
        super(ImgFeatureExtractor, self).__init__()
        self.backbone = getattr(models ,model_name)(pretrained=True)
        self.embedding  = nn.Linear(1000,512)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x
    
class TabularFeatureExtractor(nn.Module):
    def __init__(self):
        super(TabularFeatureExtractor, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features=17, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        return x
    

class ClassificationModel(nn.Module):
    def __init__(self,model_name):
        super(ClassificationModel, self).__init__()
        self.img_feature_extractor = ImgFeatureExtractor(model_name)
        self.tabular_feature_extractor = TabularFeatureExtractor()
        self.classifier = nn.Sequential(
           nn.LeakyReLU(),
           nn.Linear(in_features=1024, out_features=2)
        )
        
    def forward(self, img, tabular):
        img_feature = self.img_feature_extractor(img)
        tabular_feature = self.tabular_feature_extractor(tabular)
        feature = torch.cat([img_feature, tabular_feature], dim=-1)
        output = self.classifier(feature)
        return output
