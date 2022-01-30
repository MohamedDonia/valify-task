
"""
Created on Sun Jan 23 18:58:35 2022

@author: Mohamed Donia
"""


import sys 
import os
import math
import gc
import json
from utils import colorstr, LOGGER
sys.path.append(os.path.abspath("pytorch-image-models"))
from timm import create_model
import pandas as pd
import cv2
import random 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.squeezenet import squeezenet1_0
from torchsummary import summary
from torchviz import make_dot
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from albumentations import (Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, Resize, 
                            ChannelShuffle, RandomCrop, HueSaturationValue)
from albumentations.augmentations.transforms import Blur, CoarseDropout
from fvcore.nn import FlopCountAnalysis



seed=365
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True



def MixedUp(img1, img2, alpha):
    if img2.shape != img1.shape:
        h, w, _ = img1.shape
        img2 = cv2.resize(img2, dsize=(h, w))
        img  = img1 * (1 - alpha) + alpha * img2
    return img.astype(int)


class CustomDataGenerator(torch.utils.data.Dataset):
    
    def __init__(self, df, dim, train_flag = True):
        super(CustomDataGenerator, self).__init__()
        'Initialization'
        self.df = df
        self.dim = dim
        self.train_flag = train_flag

        # define data augmentation for train 
        self.transform_train = Compose([
            # rotate 
            Rotate(limit=10, p=random.uniform(0.4, 0.6)),
            # horizontal flip
            HorizontalFlip(p=random.uniform(0.5, 0.7)),
            # random brigthtness
            RandomBrightnessContrast(p=random.uniform(0.4, 0.6)),
            # channel shuffle
            ChannelShuffle(p=np.random.uniform(0.3, 0.5)),
            HueSaturationValue(),
            Blur(always_apply=False, p=0.5, blur_limit=(3, 7)),
            CoarseDropout(always_apply=False, p=0.5, max_holes=10, max_height=30, max_width=30, 
                          min_holes=8, min_height=10, min_width=10),
            Resize(always_apply=True, 
                   height=self.dim[0], 
                   width=self.dim[1], 
                   interpolation=0)
            ])
        # define augmentation for validation 
        self.transform_val = Compose([
            Resize(always_apply=True, 
                   height=self.dim[0], 
                   width=self.dim[1], 
                   interpolation=0)
            ])
        
        self.backgrounds = os.listdir(r'data/random background')
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.df)
    
       
    
    def __getitem__(self, idx):
        'Generates data containing batch_size samples'       
        # generate data :
        img = cv2.imread(self.df.iloc[idx, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        background_img = cv2.imread('./data/random background/' + random.choice(self.backgrounds))
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
            
        img = MixedUp(img, background_img, 0.12)    
            
        #img = cv2.resize(img, self.dim[:2])
        img = np.array(img, dtype='uint8')
        # data augmentation 
        img = self.__image_augmentation(img)
        img = img / 255.0 
        # transpose axis 
        img = img.transpose(2,1,0)
        
        y = self.df.iloc[idx, 1]
            
        return torch.tensor(img),  torch.tensor(y)
    
    def __image_augmentation(self, img):
        if self.train_flag:
            return self.transform_train(image=img)['image']
        else:
            return self.transform_val(image=img)['image']

class CheckpointCallback():
    def __init__(self,model_filename, mode='max',verbose=0) :
        self.model_filename=model_filename
        self.mode=mode
        self.verbose=verbose
        if(mode=='max'):
            self.value=-1e9
        else:
            self.value=1e9
    def check_and_save(self,model:torch.nn.Module,value):
        save=False
        if(self.mode =='max'):
            if(value>self.value):
                if(self.verbose==1):
                    print(colorstr('blue', 'bold', f'\n model saved with value {value:.3f} previous is {self.value:.3f}'))
                self.value=value
                save=True
        if(self.mode == 'min'):
            if(value<self.value):
                if(self.verbose==1):
                    print(colorstr('blue', 'bold', f'\n model saved with value {value:.3f} previous is {self.value:.3f}'))
                self.value=value
                save=True
        if(save):
            torch.save(model.state_dict(),self.model_filename)
        return 
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                

class MyTrain():
    def __init__(self, model, train_loader, val_loader, loss, 
                 optimizer, ckpt_callback, early_stopping, epochs, scheduler):
        self.MODEL = model
        self.TRAIN_LOADER = train_loader
        self.VAL_LOADER = val_loader
        self.LOSS = loss
        self.OPTIMIZER = optimizer
        self.CKPT_CALLBACK = ckpt_callback
        self.EPOCHS = epochs
        self.EARLY_STOPPING = early_stopping
        self.SCHEDULAR = scheduler
        self.BEST_Acc = 0
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.logs = {}
        
        
        
    def run(self):
        for e in range(self.EPOCHS):
            LOGGER.info(('%12s' * 5) % ('Epoch', 'gpu_mem', 'LR','loss', 'Acc'))
            pbar=tqdm(self.TRAIN_LOADER, total=len(self.TRAIN_LOADER), position=0, leave=True)
            acc_loss = 0
            acc_acc = 0
            self.MODEL.train()
            for i, (img_batch, y) in enumerate(pbar):
                # data loading :
                img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.long)
                
                # predict = forward pass with our model
                y_predicted = self.MODEL(img_batch)
                # calculate loss 
                l = self.LOSS(y_predicted, y)
                acc = Accuracy(y_predicted, y)
                
                # accumlated loss 
                acc_loss = (acc_loss * i + l.item()) / (i + 1)  # update mean losses
                # accumlated accuracy
                acc_acc = (acc_acc * i + acc.item()) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%12s' * 2 + '%12.4g' * 3) % (f'{e+1}/{self.EPOCHS}', mem, 
                                                                    self.SCHEDULAR.get_last_lr()[0], acc_loss, acc_acc))
                
                
                # calculate gradients = backward pass
                l.backward()
                # update weights
                self.OPTIMIZER.step()
                # zero the gradients after updating
                self.OPTIMIZER.zero_grad()
                # schedular step
                self.SCHEDULAR.step()
            self.train_loss.append(acc_loss)
            self.train_acc.append(acc_acc)
            
            # evaluation 
            self.MODEL.eval()
            acc_loss = 0
            acc_acc = 0
            with torch.no_grad():
                LOGGER.info(('%12s' * 4) % ('', 'gpu_mem', 'LR', 'loss'))
                pbar=tqdm(self.VAL_LOADER, total=len(self.VAL_LOADER), position=0, leave=True)
        
                for j, (img_batch, y) in enumerate(pbar):
                    # data loading :
                        img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                        y = y.to(DEVICE, dtype=torch.long)
                        # predict = forward pass with our model
                        y_predicted = self.MODEL(img_batch)
                        l = self.LOSS(y_predicted, y)
                        acc = Accuracy(y_predicted, y)
                        # accumlated loss 
                        acc_loss = (acc_loss * j + l.item()) / (j + 1)  # update mean losses
                        # accumlated accuracy
                        acc_acc = (acc_acc * j + acc.item()) / (j + 1)  # update mean losses
                        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                        pbar.set_description(('%12s' * 2 + '%12.4g' * 3) % ('     ',mem, 
                                                                            self.SCHEDULAR.get_last_lr()[0], acc_loss, acc_acc))
            
            self.val_loss.append(acc_loss)
            self.val_acc.append(acc_acc)
            
            # save model 
            self.CKPT_CALLBACK.check_and_save(self.MODEL, acc_acc)
            if acc_acc > self.BEST_Acc:
                self.BEST_Acc = acc_acc
            # early stopping     
            self.EARLY_STOPPING(acc_loss)
            if self.EARLY_STOPPING.early_stop:
                break
        # save logs of traing 
        self.logs['train loss'] = self.train_loss
        self.logs['train acc'] = self.train_acc
        self.logs['val loss'] = self.val_loss
        self.logs['val acc'] = self.val_acc
            

def Accuracy(preds, target):
    winners = torch.softmax(preds, dim=1).argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy



IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS=20
LR=1e-4
DEVICE = 'cuda:1'
NFOLDS = 5



df = pd.read_csv('data.csv')
df = df.sample(frac=1).reset_index(drop=True)
skf = StratifiedKFold(n_splits=NFOLDS, random_state = seed, shuffle=True)


best_Acc = []
folds_logs = {}
for i, (train_df_index, val_df_index) in enumerate(skf.split(df.index, df['label'])):
    
    print(colorstr('blue', 'bold', f'****************** Fold {i+1} *********************'))
    train_df = df.iloc[train_df_index]
    val_df   = df.iloc[val_df_index]

    
    train_data = CustomDataGenerator(train_df, dim=IMG_SIZE, train_flag=True)
    val_data   = CustomDataGenerator(val_df,   dim=IMG_SIZE, train_flag=False)
    


    train_loader=DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=30,
                            pin_memory=True)

    val_loader=DataLoader(dataset=val_data,
                         batch_size=BATCH_SIZE*2,
                         shuffle=False,
                         num_workers=30,
                         pin_memory=True)

    print(colorstr('blue', 'bold', 'DataLoading .....Done!'))
   
    
   
    
   
    # models : 
    # resnet 18    
    '''
    model = create_model(model_name = 'resnet18', 
                         pretrained=True,
                         num_classes = 29)
    summary(model.to('cuda:0'), input_size=(3,*IMG_SIZE))
    '''
    # mobilenet
    '''
    model = create_model(model_name = 'mobilenetv2_100', 
                         pretrained=True,
                         num_classes = 29)
    '''
    # squeeze net 1.0

    model = squeezenet1_0(pretrained=False,
                         num_classes = 29)
    # squeeze net 1.1
    '''
    model = squeezenet1_1(pretrained=False,
                         num_classes = 29)
    '''
    

    
    model.to(DEVICE)



    print(colorstr('blue', 'bold', 'Creating Model .....Done!'))
    # summary(model.to(DEVICE), input_size=(3,*IMG_SIZE))

    loss = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = OneCycleLR(optimizer, 
                           max_lr=LR, 
                           steps_per_epoch=len(train_loader), 
                           epochs=EPOCHS)
    
    ckpt_callback = CheckpointCallback(f'squeezenet1_0_{i+1}.pt', 'max', verbose=1)
    early_stopping = EarlyStopping(patience=3)
    train_1 = MyTrain(model, 
                      train_loader, 
                      val_loader, 
                      loss, 
                      optimizer, 
                      ckpt_callback, 
                      early_stopping, 
                      EPOCHS, scheduler)
    
    train_1.run()
    best_Acc.append(train_1.BEST_Acc)
    folds_logs[f'fold {i+1}'] = train_1.logs
    
    
    # delete model :
    del train_1
    model.to('cpu')
    del model
    gc.collect()
    with torch.cuda.device(DEVICE): 
         torch.cuda.empty_cache()
         
    break     
    
with open('squeezenet 1_0 training logs.json', 'w') as file:
    json.dump(folds_logs, file)
    
    