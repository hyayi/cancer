from torch import nn
import models.backbone as backbone
import models.aggregater as aggregater
import models.classify  as classify


class MILNetImage(nn.Module):
    """Training with image only"""

    def __init__(self,model_config):
        super().__init__()

        self.image_feature_extractor = getattr(backbone, model_config['backbone_name'])(**model_config['backbone_params'])
        self.aggregator = getattr(aggregater, model_config['aggregator_name'])(input_size = self.image_feature_extractor.output_size, **model_config['aggregator_params'])
        self.classifier = getattr(classify, model_config['classifyr_name'])(input_size = self.aggregator.output_size, **model_config['classify_params'])

    def forward(self, bag_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature = self.aggregator(patch_features)
        result = self.classifier(aggregated_feature)
        return result

    
class MILNetImageMLP(nn.Module):
    """Training with image only"""

    def __init__(self,model_config):
        super().__init__()

        self.image_feature_extractor = getattr(backbone, model_config['backbone_name'])(**model_config['backbone_params'])
        self.classifier = nn.Sequential(
            nn.Linear(self.image_feature_extractor.output_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, bag_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        result = self.classifier(patch_features)
        return result
