import random
import numpy as np
import os

import torch

from torch.utils.data import  DataLoader
from lightning_model import BreastClassfication
import warnings
from dataset import BreastDataset
import pytorch_lightning as pl
import argparse
import yaml
from transforms import get_transforms

warnings.filterwarnings(action='ignore') 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(41)

def train(config):
    
    train_transforms, test_transforms = get_transforms()
    

    train_dataset = BreastDataset(config.train_path, config.data_dir_path,is_preloading=False,transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True, num_workers=config.num_workers,pin_memory=config.pin_memory)
    
    print(len(train_dataset))

    val_dataset = BreastDataset(config.val_path, config.data_dir_path,is_preloading=False,transforms=test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers,pin_memory=config.pin_memory)
    
    with open(config.model_config_path) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
        
    model = BreastClassfication(learning_rate = config.learning_rate,model_config = model_config)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"{config.weight_path}/", save_top_k=1, monitor="val_f1",filename=f'{config.model_name}'+'-{epoch:02d}-{val_f1:.3f}',mode='max')
    callbacks = [checkpoint_callback]
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=f"{config.log_path}/")
    
    trainer = pl.Trainer(accelerator=config.accelerator, devices=config.device, precision=16,max_epochs=config.epoch,callbacks=callbacks,strategy=config.strategy,logger=tb_logger,gradient_clip_val=config.gradient_clip_val)
    
    trainer.fit(model,train_dataloaders = train_loader, val_dataloaders = val_loader)

    trainer.validate(model, ckpt_path=checkpoint_callback.best_model_path, dataloaders = val_loader)
    
    
    
if __name__=="__main__" :
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default='./data_patch/train.csv')
    parser.add_argument("--val_path", type=str, default='./data_patch/val.csv')
    parser.add_argument("--data_dir_path", type=str, default='./data_patch/train')
    parser.add_argument("--weight_path", type=str, default='./')
    parser.add_argument("--log_path", type=str, default='./')
    parser.add_argument("--model_config_path", type=str, default='./model_config/experiment2.yaml')
    parser.add_argument("--model_name", type=str, default='experiment1')
    
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--pin_memory",type=bool, default=False)
    parser.add_argument("--epoch",type=int, default=10)
    parser.add_argument("--learning_rate",type=float, default=0.0001)
    
    parser.add_argument("--accelerator",type=str, default='gpu')
    parser.add_argument("--device",type=int,default=1)
    parser.add_argument("--gradient_clip_val",type=float, default=None)
    parser.add_argument("--strategy",type=str, default=None)
    
    config = parser.parse_args()
    
    train(config=config)