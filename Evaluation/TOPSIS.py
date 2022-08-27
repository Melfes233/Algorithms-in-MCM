'''
Python实现TOPSIS分析法（优劣解距离法）
https://blog.csdn.net/weixin_41799019/article/details/97611462
https://zhuanlan.zhihu.com/p/37738503
'''
import numpy as np
import pandas as pd

def dataDirection_1(datas):
    #极小型指标转成极大型指标
    return 1.0/datas
    return np.max(datas)-datas

def dataDirection_2(datas,x_best):
    #中间型指标转成极大型指标
    tmp_datas=abs(datas-x_best)
    Max=np.max(tmp_datas)
    return np.ones_like(datas)-abs(tmp_datas)/Max

def dataDirection_3(datas,low,up):
    #区间型指标转成极大型指标(最佳区间[a,b]),待改进
    Max=max(low-np.min(datas),np.max(datas)-up)
    res=np.zeros_like(datas)
    for i in range(len(datas)):
        if datas[i]<low:
            res[i]=1-(low-datas[i])*1.0/Max
        elif low<=datas<=up:
            res[i]=datas[i]
        else:
            res[i]=1-(datas[i]-up)*1.0/Max
    return res

def normalize(datas):
    #正向化矩阵标准化(列标准化)
    # print(datas)
    norm=np.power(np.sum(pow(datas,2),axis=0),0.5)#每列平方和再开根号
    # print(pow(datas,2))
    # print(norm)
    for i in range(norm.shape[0]):
        datas[i]=datas[i]/norm[i]
    return datas

def topsis(normed_data,weight):
    #最优最劣方案
    Z=pd.DataFrame([normed_data.min(),normed_data.max()],index=['negative','positive'])
    
    #距离
    score=[]
    for i in range(normed_data.shape[0]):
        dis_pos=np.sqrt(((normed_data-Z.loc('positive'))**2*weight).sum(axis=1))
        dis_neg=np.sqrt(((normed_data-Z.loc('negative'))**2*weight).sum(axis=1))
        score[i]=dis_neg/(dis_pos+dis_neg)
    
    return score

def main(data,weight,data_type):
    data_processed=np.ones_like(data)
    for i in range(len(data_type)):
        if data_type[i]==0:
            data_processed[:,i]=data[:,i]
        elif data_type[i]==1:
            data_processed[:,i]=dataDirection_1(data[:,i])
        elif data_type[i]==2:
            data_processed[:,i]=dataDirection_2(data[:,i],x_best=1)
        elif data_type[i]==3:
            data_processed[:,i]=dataDirection_3(data[:,i],low=5,up=6)
    print(data_processed)
    data_normed=normalize(data_processed)
    print(data_normed)
        
    

if __name__=='__main__':
    # row=int(input('行数(样本数):'))
    # col=int(input('列数(指标数):'))
    # data=np.array((row,col),dtype=float)
    '''
    读取数据
    '''
    weight=np.array([0.2,0.3,0.4,0.1])
    data=np.array([[0.1,5,5000,4.7],
                   [0.2,6,6000,5.6],
                   [0.4,7,7000,6.7],
                   [0.9,10,10000,2.3],
                   [1.2,2,400,1.8]
                    ],dtype=float)
    data_type=[0,2,0,1]
    main(data=data,weight=weight,data_type=data_type)
    
            
            
                

    