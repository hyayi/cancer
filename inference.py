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
import torch.nn.functional as F

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

def inference(config):

    _, _, test_df, _, _ = preprocessing(config.train_path, config.test_path)

    _,test_transforms =  get_transforms(h =config.img_h, w=config.img_w)

    test_dataset = CustomDataset(test_df, None, test_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,pin_memory=config.pin_memory)

    model = MRSClassfication.load_from_checkpoint(config.weight_path, learning_rate = config.learning_rate, model_name=config.model_name)
    model.to(config.device)
    model.eval()
    preds = []
    
    with torch.no_grad():
        for img, tabular in iter(test_loader):
            img = img.to(config.device)
            tabular = tabular.to(config.device)
            model_pred = model(img, tabular)
            model_pred = F.softmax(model_pred.detach().cpu(), dim=-1)
            preds += torch.argmax(model_pred,dim=-1).tolist()
            
    test_df['N_category'] = preds
    submit = test_df[['ID','N_category']]
    submit.to_csv(f'{config.save_path}/{config.model_name}_submit.csv', index=False)
    
if __name__=="__main__" :
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default='./train_re.csv')
    parser.add_argument("--test_path", type=str, default='./test_re.csv')
    parser.add_argument("--weight_path", type=str, default='eb7-epoch=07-val_f1=0.77.ckpt')
    parser.add_argument("--model_name", type=str, default='eb7')
    parser.add_argument("--save_path", type=str, default='./')
    
    parser.add_argument("--img_h", type=str, default=512)
    parser.add_argument("--img_w", type=str, default=512)
    
    parser.add_argument("--batch_size",type=int, default=4)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--pin_memory",type=bool, default=False)
    parser.add_argument("--learning_rate",type=float, default=0.0001)
    parser.add_argument("--device",type=str, default='cuda')
    config = parser.parse_args()
    
    inference(config=config)
