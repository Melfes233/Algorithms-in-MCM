import numpy as np
from scipy.sparse.linalg import eigs

def getweight(n):
    matrix0=np.zeros((n,n))
    for i in range(n):
        matrix0[i,i]=1.0
        for j in range(i+1,n):
            matrix0[i,j]=float(input('第{}个比第{}个的标度:'.format(str(i+1),str(j+1))))
            matrix0[j,i]=1.0/matrix0[i,j]

    print(matrix0)
    return matrix0

def ri(n):
    score=[0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45]
    return score[n]

def AHP(matrix0,n):
    val,vec=eigs(matrix0,1)#max
    ci=(val-n)/(n-1.0)
    w=vec/sum(vec)
    print('最大特征值：',val)
    print('对应特征向量:',w)
    print('CI=',ci)
    print('RI=',ri(n))
    print('CR=',ci/ri(n))
    return w

def AHP_main():
    n=int(input('dim:'))
    matrix0=getweight(n)
    weight=AHP(matrix0,n)
    print(weight)
    return weight
    
if __name__=='__main__':
    AHP_main()