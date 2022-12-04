import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.basemodel import ClassificationModel
from torchmetrics.functional.classification import multiclass_f1_score
from schduler import CosineAnnealingWarmUpRestarts

class MRSClassfication(pl.LightningModule):

    def __init__(self,model_name,learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = ClassificationModel(model_name)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img,tabular):
        return self.model(img,tabular)

    def training_step(self, batch, batch_idx):
        img, tabular, label= batch
        pred = self(img, tabular)
        
        loss = self.loss(pred, label)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'loss':loss,'pred':pred,'label':label}
        
        return output
    
    def training_epoch_end(self, outputs):
        
       preds = torch.cat([x['pred'] for x in outputs]).view(-1,2)
       labels = torch.cat([x['label'] for x in outputs]).view(-1)
       
       f1_macro = multiclass_f1_score(preds,labels,num_classes=2)
       self.log("train_f1", f1_macro,prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        img, tabular, label= batch
        pred = self(img, tabular)
        
        loss = self.loss(pred, label)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'pred':pred,'label':label}

        return output

    def validation_epoch_end(self, outputs):
        
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
            
        preds = torch.cat([x['pred'] for x in outputs]).view(-1,2)
        labels = torch.cat([x['label'] for x in outputs]).view(-1)

        f1_macro = multiclass_f1_score(preds,labels,num_classes=2)
        self.log("val_f1", f1_macro, prog_bar=True, logger=True,rank_zero_only=True,on_epoch=True)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=self.learning_rate, T_up=10, gamma=0.5)
        return [optimizer],[scheduler]
    
    
