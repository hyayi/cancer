import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.functional.classification import multiclass_f1_score
from schduler import CosineAnnealingWarmUpRestarts
from models.pathmodel import MILNetImage

class BreastClassfication(pl.LightningModule):

    def __init__(self,model_config,learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = MILNetImage(model_config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, bag_data):
        return self.model(bag_data)

    def training_step(self, batch, batch_idx):
        pred = self(batch['bag_tensor'])
        
        loss = self.loss(pred, batch['label'])
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'loss':loss,'pred':pred,'label':batch['label']}
        
        return output
    
    def training_epoch_end(self, outputs):
        
       preds = torch.cat([x['pred'] for x in outputs])
       labels = torch.cat([x['label'] for x in outputs])
       
       f1_macro = multiclass_f1_score(preds,labels,num_classes=2)
       self.log("train_f1", f1_macro,prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pred = self(batch['bag_tensor'])
        
        loss = self.loss(pred, batch['label'])
        
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'pred':pred,'label':batch['label']}

        return output

    def validation_epoch_end(self, outputs):
        
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
            
        preds = torch.cat([x['pred'] for x in outputs])
        labels = torch.cat([x['label'] for x in outputs])
        print(preds.shape)
        print(labels.shape)
        f1_macro = multiclass_f1_score(preds,labels,num_classes=2)
        self.log("val_f1", f1_macro, prog_bar=True, logger=True,rank_zero_only=True,on_epoch=True)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=self.learning_rate, T_up=10, gamma=0.5)
        return [optimizer],[scheduler]
