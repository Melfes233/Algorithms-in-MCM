'''
Python实现TOPSIS分析法（优劣解距离法）
https://blog.csdn.net/weixin_41799019/article/details/97611462
'''
import numpy as np

def dataDirection_1(datas):
    #极小型指标转成极大型指标
    return np.max(datas)-datas

def dataDirection_2(datas,x_best):
    #中间型指标转成极大型指标
    tmp_datas=abs(datas-x_best)
    Max=np.max(tmp_datas)
    return np.ones_like(datas)-abs(tmp_datas)/Max

def dataDirection_3(datas,x_best,low,up):
    #区间型指标转成极大型指标(最佳区间[a,b])
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
    norm=np.power(np.sum(pow(datas,2),axis=1),0.5)#每列平方和再开根号
    for i in range(norm.size()):
        datas[i]=datas[i]/norm[i]
    return datas


            

    