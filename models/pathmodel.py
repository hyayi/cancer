from torch import nn
import models.backbone as backbone
import models.aggregater as aggregater
import models.classify  as classify
import models.tabular  as tabular
import torch 

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

class MILNetMultimodal(nn.Module):
    """Training with image and tabular data"""

    def __init__(self,model_config):
        super().__init__()
        self.contrastive_loss = model_config['contrastive_loss']
        self.image_feature_extractor = getattr(backbone, model_config['backbone_name'])(**model_config['backbone_params'])
        self.aggregator = getattr(aggregater, model_config['aggregator_name'])(input_size = self.image_feature_extractor.output_size, **model_config['aggregator_params'])
        self.clinical_feature_extractor = getattr(tabular, model_config['clinical_feature_extractor_name'])(**model_config['clinical_feature_extractor_params'])
        self.classifier = getattr(classify, model_config['classifyr_name'])(input_size = self.aggregator.output_size + self.clinical_feature_extractor.output_size, **model_config['classify_params'])

    def forward(self, bag_data, tabular_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature = self.aggregator(patch_features)
        clinical_features = self.clinical_feature_extractor(tabular_data)
        total_featrues = torch.cat([aggregated_feature, clinical_features], dim=1)
        result = self.classifier(total_featrues)
        
        if self.contrastive_loss:
            return result, aggregated_feature, clinical_features
        else:
            return result