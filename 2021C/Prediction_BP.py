import sys
import torch
import torch.nn as nn
import os
import math
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import random
from sklearn.model_selection import KFold

class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP,self).__init__()
        #model structure
        self.net=nn.Sequential(
            nn.Linear(input_dim,12),
            nn.Linear(12,4),
            nn.Linear(4,1),
        )

        
    def forward(self,x):
        x=self.net(x)     
        return x
    
def trainer(train_loaders,valid_loader,model,config,device):
    #loss function
    lossfunc=nn.MSELoss(reduction='mean')
    #optimizer
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    
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
            # print(x)
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
    'epoches': 300,     # Number of epochs.            
    'batch_size': 1, 
    'lr': 1e-4, 
    'weight_decay': 1e-3,       
    'early_stop': 20,    # If model has not improved for this many consecutive epochs, stop training.     
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
            # pred = pred.detach().cpu()                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()
    return preds,goal

def read_xl(input_path):
    workbook=pd.read_excel(input_path,usecols=range(1,242))
    data=workbook.values
    return data


def dataloader(data,input_dim,epoches):
    test_data=[]
    for i in range(217,240):
        test_data.append((torch.tensor(data[i-input_dim:i].tolist(),dtype=torch.float32),torch.tensor(data[i],dtype=torch.float32)))
    # test_data=torch.tensor(test_data)
 
    train_data=[]
    for i in range(input_dim,217):
        train_data.append({'data':data[i-input_dim:i].tolist(),'goal':data[i]})
    # train_data=torch.tensor(train_data)
    # print(test_data)

    def loader(data,epoches=epoches):
        for _ in range(epoches):
            data_combined=[]
            random.shuffle(data)
            for k in data:
                his=torch.tensor(k['data'],dtype=torch.float32)
                goal=torch.tensor([k['goal']],dtype=torch.float32)
                data_combined.append((his,goal))
            yield data_combined
    return loader(train_data),test_data



if __name__=='__main__':

    input_dim=48
    input_path=os.path.join(sys.path[0],'data','data_chosen.xlsx')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    total_data=read_xl(input_path)
    for i in range(total_data.shape[0]):

        data=total_data[i][1:]
        name=total_data[i][0]
        
        avg,sigma=np.average(data),np.std(data)
        data=(data-avg)/sigma
        
        train_loader,test_loader=dataloader(data,input_dim=input_dim,epoches=config['epoches'])
        config['save_path']='./2021C/models/model_{}.ckpt'.format(name)
        print(name,data[0])
        model = MLP(input_dim=input_dim).to(device) # put your model and data on the same computation device.
        # model.train()
        # ans=model(torch.tensor([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000],dtype=torch.float32).to(device))
        # print(ans)
        '''
        '''
        trainer(train_loader, test_loader, model, config, device)
        
        #pred
        model = MLP(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(config['save_path']))

        preds,goal= predict(test_loader, model, device)
        mseloss=0.5*np.sum((preds-goal)**2)
        preds=preds.tolist()
        for i in range(len(goal)):
            goal[i]=float(goal[i])
        test={}
        tmp=[]
        for i in range(len(preds)):
            pred_relu=preds[i]*sigma+avg
            if pred_relu<0:
                pred_relu=0
            tmp.append({'pred':pred_relu,'goal':goal[i]*sigma+avg})
        test['test']=tmp
        test['test_loss']=mseloss    

        data=data.tolist()
        preds = []
        for i in range(48): 
            # print('{}   {}\n'.format(preds[i]*sigma+avg,goal[i]*sigma+avg))
            x=torch.tensor(data[-48:],dtype=torch.float32)
            x = x.to(device)    
            model.eval() # Set your model to evaluation mode.
            with torch.no_grad():                   
                pred = model(x)                     
                preds.append(pred.detach().cpu())
                data.append(preds[-1])   
        preds = torch.cat(preds, dim=0).numpy()  
        test['prediction']=[]
        for i in range(48):
            pred_relu=preds[i]*sigma+avg
            if pred_relu<0:
                pred_relu=0
            test['prediction'].append({'week':241+i,'pred':float(pred_relu)})

        with open(os.path.join(sys.path[0],'models','output','chosen','{}.json'.format(name)),'w',encoding='utf-8') as f:
            json.dump(test,f,ensure_ascii=False)