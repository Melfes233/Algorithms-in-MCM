import numpy as np
import matplotlib.pyplot as plt

def getweight(n):
    matrix0=np.zeros((n,n))
    for i in range(n):
        matrix0[i,i]=0.5
        for j in range(i+1,n):
            matrix0[i,j]=float(input('第{}个比第{}个的标度:'.format(str(i+1),str(j+1))))
            matrix0[j,i]=1.0-matrix0[i,j]

    print(matrix0)

    weight=np.ones(n)

    for i in range(n):
        sum_row=np.sum(matrix0,axis=1)
        weight[i]=(sum_row[i]+n/2.0-1)/(n*(n-1.0))

    return weight





if __name__=='__main__':
    print(getweight(3))
