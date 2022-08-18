'''
需要根据题目搜集和处理数据并写一个dataloader,数据集以np数组存储
'''

import torch
import torch.nn as nn
import os
import math
from tqdm import tqdm
import numpy as np

class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP,self).__init__()
        #model structure
        self.net=nn.Sequential(
            nn.Linear(input_dim,32),
            nn.Relu(),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.ReLU()
        )
        
    def forward(self,x):
        x=self.net(x)
        x=x.squeeze(1)#(dim,1)->(dim)
        return x
    
def trainer(train_loader,valid_loader,model,config,device):
    #loss function
    lossfunc=nn.MSELoss(reduction='mean')
    #optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    
    epoches,best_loss,step,early_stop_count=config['epoches'],math.inf,0,0
    
    for epoch in range(epoches):
        model.train()
        loss_record=[]
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'epoches': 5000,     # Number of epochs.            
    'batch_size': 128, 
    'lr': 1e-5, 
    'weight_decay': 1e-4,       
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
} 

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

if __name__=='__main__':
    
    model = MLP(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
    trainer(train_loader, valid_loader, model, config, device)
    
    #pred
    model = MLP(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device) 