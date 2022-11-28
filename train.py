import random
import numpy as np
import os

import torch

from torch.utils.data import  DataLoader


from sklearn.model_selection import train_test_split
from linghtning_model import MRSClassfication
import warnings
from dataset import CustomDataset
from transforms import get_transforms
from utils import preprocessing
import pytorch_lightning as pl
import argparse

warnings.filterwarnings(action='ignore') 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def train(config):

    train_df, val_df, test_df, train_labels, val_labels = preprocessing(config.train_path, config.test_path)

    
    train_transforms,test_transforms =  get_transforms(config.img_size)
    train_dataset = CustomDataset(train_df, train_labels.values, train_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=config.num_workers,pin_memory=config.pin_memory)

    val_dataset = CustomDataset(val_df, val_labels.values, test_transforms)
    
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,pin_memory=config.pin_memory)

    model = MRSClassfication(learning_rate = config.learning_rate)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"{config.weight_path}/", save_top_k=1, monitor="val_f1",filename=f'{config.name}'+'-{epoch:02d}-{val_f1:.2f}',mode='max')
    
    callbacks = [checkpoint_callback]
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=f"{config.log_path}/")
    
    trainer = pl.Trainer(accelerator=config.accelerator, devices=config.device, precision=16,max_epochs=config.epoch,callbacks=callbacks,strategy=config.strategy,logger=tb_logger,gradient_clip_val=config.gradient_clip_val)
    
    trainer.fit(model,train_dataloaders = train_loader, val_dataloaders = val_loader)
    
    trainer.validate(model, ckpt_path=checkpoint_callback.best_model_path, dataloaders = val_loader)
    
    
    
if __name__=="__main__" :
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default='./train_re.csv')
    parser.add_argument("--test_path", type=str, default='./test_re.csv')
    parser.add_argument("--weight_path", type=str, default='./')
    parser.add_argument("--log_path", type=str, default='./')
    parser.add_argument("--name", type=str, default='res50')
    
    
    parser.add_argument("--img_size", type=str, default=512)
    parser.add_argument("--batch_size",type=int, default=4)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--pin_memory",type=bool, default=False)
    parser.add_argument("--epoch",type=int, default=50)
    parser.add_argument("--learning_rate",type=float, default=0.0001)
    
    parser.add_argument("--accelerator",type=str, default='gpu')
    parser.add_argument("--device",type=int,default=1)
    parser.add_argument("--gradient_clip_val",type=float, default=None)
    parser.add_argument("--strategy",type=str, default=None)
    
    config = parser.parse_args()
    
    train(config=config)