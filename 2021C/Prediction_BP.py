'''
需要根据题目搜集和处理数据并写一个dataloader,数据集以np数组存储
'''
import sys
import torch
import torch.nn as nn
import os
import math
from tqdm import tqdm
import numpy as np
import xlwt
import pandas as pd
import random
from sklearn.model_selection import KFold

class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP,self).__init__()
        #model structure
        self.net=nn.Sequential(
            nn.Linear(input_dim,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.ReLU(),
        )
        
    def forward(self,x):
        x=self.net(x)
        # print(x)
        # x=x.squeeze(1)#(dim,1)->(dim)
        return x
    
def trainer(train_loaders,valid_loaders,model,config,device):
    #loss function
    lossfunc=nn.MSELoss(reduction='mean')
    #optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    
    if not os.path.isdir('./2021C/models'):
        os.mkdir('./2021C/models')
    
    epoches,best_loss,step,early_stop_count=config['epoches'],math.inf,0,0
    
    # for epoch in range(epoches):
    epoch=-1
    for train_loader in train_loaders:
        epoch=epoch+1


        loss_record=[]
        model.train()
        train_bar=tqdm(train_loader,position=0,leave=True)
        
        for x,y in train_bar:
            optimizer.zero_grad()
            x,y=x.to(device),y.to(device)
            pred=model(x)
            loss=lossfunc(pred,y)
            loss.backward()
            optimizer.step()
            step+=1
            loss_record.append(loss.detach().item())
            
            
            train_bar.set_description('Epoch {}/{}'.format((epoch+1),epoches))
            train_bar.set_postfix({'loss':loss.detach().item()})
        # print(len(loss_record))
        mean_train_loss=sum(loss_record)/len(loss_record)
        
        model.eval()
        loss_record=[]
        for x,y in valid_loader:
            x,y=x.to(device),y.to(device)
            with torch.no_grad():
                pred=model(x)
                loss=lossfunc(pred,y)
                
            loss_record.append(loss.item())
            
        mean_valid_loss=sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{epoches}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        
        if mean_valid_loss<best_loss:
            best_loss=mean_valid_loss
            torch.save(model.state_dict(),config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
            
        

config = {
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'epoches': 10,     # Number of epochs.            
    'batch_size': 1, 
    'lr': 1e-5, 
    'weight_decay': 1e-4,       
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './2021C/models/model.ckpt'  # Your model will be saved here.
} 

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    goal=[]
    for x,y in tqdm(test_loader):
        x = x.to(device)    
        goal.append(y)                    
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds,goal

def read_xl(input_path,row=0):
    workbook=pd.read_excel(input_path,usecols=range(1,242))
    data=workbook.values
    return data[row][1:],data[row][0]


def dataloader(input_path,input_dim,epoches,epochn_split=5):
    data,name=read_xl(input_path)

    test_data=[]
    for i in range(217,240):
        test_data.append((torch.tensor(data[i-input_dim:i].tolist()),torch.tensor(data[i])))
    # test_data=torch.tensor(test_data)

    valid_data=[]
    for i in range(192,217):
        valid_data.append((torch.tensor(data[i-input_dim:i].tolist()),torch.tensor(data[i])))
    # valid_data=torch.tensor(valid_data)
    
    train_data=[]
    for i in range(input_dim,192):
        train_data.append({'data':data[i-input_dim:i].tolist(),'goal':data[i]})
    # train_data=torch.tensor(train_data)
    # print(test_data)

    def loader(data,epoches=epoches):
        for _ in range(epoches):
            data_combined=[]
            random.shuffle(data)
            for k in data:
                his=torch.tensor(k['data'])
                goal=torch.tensor(k['goal'])
                data_combined.append((his,goal))
            yield data_combined
    return loader(train_data),test_data,valid_data,name
    
    '''
    # kf=KFold(n_splits=n_split)
    
    # for train_index,valid_index in kf.split(train_set[input_dim:]):
        # train_data=train_data[train_index]
        # valid_data=train_data[valid_index]
        # print('train:\n',train_data)
        # print('test:\n',test_data)
        # print(train_index)
        train_data=[]
        for i in train_index:
            train_data.append(train_set[i:i+input_dim])
        train_data=torch.tensor(train_data)
        valid_data=[]
        for i in valid_index:
            train
    '''
        
        



if __name__=='__main__':

    input_dim=12
    input_path=os.path.join(sys.path[0],'data','F50_data.xlsx')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    train_loader,test_loader,valid_loader,name=dataloader(input_path,input_dim=input_dim,epoches=config['epoches'])

    model = MLP(input_dim=input_dim).to(device) # put your model and data on the same computation device.
    trainer(train_loader, valid_loader, model, config, device)
    
    #pred
    model = MLP(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds,goal= predict(test_loader, model, device)
    for i in range(len(preds)): 
        print('{}   {}\n'.format(preds[i],goal[i]))