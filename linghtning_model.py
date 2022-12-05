import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.basemodel import ClassificationModel
from torchmetrics.functional.classification import multiclass_f1_score
from schduler import CosineAnnealingWarmUpRestarts
from torch import einsum

class MRSClassfication(pl.LightningModule):

    def __init__(self,model_name,learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = ClassificationModel(model_name)
        self.loss = nn.CrossEntropyLoss()
        self.temperature = nn.Parameter(torch.Tensor([1.]))

    def forward(self, img,tabular):
        return self.model(img,tabular)

    def training_step(self, batch, batch_idx):
        img, tabular, label= batch
        pred,img_em,tabular_em = self(img, tabular)
        
        sim = einsum('i d, j d -> i j', tabular_em, img_em)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(img.shape[0],dtype=torch.int).to(label)
        contrastive_loss = (self.loss(sim, contrastive_labels) + self.loss(sim.t(), contrastive_labels)) * 0.5
        loss = self.loss(pred, label) + contrastive_loss
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'loss':loss,'pred':pred,'label':label}
        
        return output
    
    def training_epoch_end(self, outputs):
        
       preds = torch.cat([x['pred'] for x in outputs]).view(-1,2)
       labels = torch.cat([x['label'] for x in outputs]).view(-1)
       
       f1_macro = multiclass_f1_score(preds,labels,num_classes=2)
       self.log("train_f1", f1_macro,prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        img, tabular, label= batch
        pred,img_em,tabular_em = self(img, tabular)
        
        sim = einsum('i d, j d -> i j', tabular_em, img_em)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(img.shape[0],dtype=torch.int).to(label)
        contrastive_loss = (self.loss(sim, contrastive_labels) + self.loss(sim.t(), contrastive_labels)) * 0.5
        loss = self.loss(pred, label) + contrastive_loss
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'pred':pred,'label':label}

        return output

    def validation_epoch_end(self, outputs):
        
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
            
        preds = torch.cat([x['pred'] for x in outputs]).view(-1,2)
        labels = torch.cat([x['label'] for x in outputs]).view(-1)

        f1_macro = multiclass_f1_score(preds,labels,num_classes=2)
        self.log("val_f1", f1_macro, prog_bar=True, logger=True,on_epoch=True)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=self.learning_rate, T_up=10, gamma=0.5)
        return [optimizer],[scheduler]
    
    
