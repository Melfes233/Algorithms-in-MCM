'''
假设现在有这么一个函数：
f(x)=x^3-60x^2-ax+6
现要求其在[0,100]范围内的最小值
https://finthon.com/python-simulated-annealing/
'''

import numpy as np
import matplotlib.pyplot as plt
import math

def func(x):
    return (x**3-60*x**2-4*x+6)

def show(x,y):
    plt.plot(x,y)
    plt.show()

def prob(i,j,func,k,T):
    e_i=func(i)
    e_j=func(j)
    T=T*1.0
    if e_j<e_i:
        return 1
    else:
        return math.exp((e_i-e_j)/(k*T))

def x_is_right(x):
    return (0<=x<=100)

def SA(function,#能量函数
       x,#初始输入
       x_is_right,#判断所得x是否符合定义区间
       T_init=1000,#初始温度
       T_min=1,#终止温度
       a=0.99,#衰减率
       k=1,#计算概率时的k值
       ):
    T=T_init
    while T>T_min:
        y=function(x)
        #新值通过扰动产生
        x_new=x+np.random.uniform(-1,1)
        if x_is_right(x_new):
            y_new=function(x_new)
            random_num=np.random.uniform(0,1)
            p=prob(x,x_new,function,k,T)
            if random_num<=p:
                x=x_new
        T=T*a
    return x
        
       
if __name__=='__main__':
    x=np.linspace(0,100,1000)
    y=func(x)
    show(x,y)
    x_ans_avg=0
    total_num=100
    for i in range(total_num):
        x_init=np.random.uniform(0,100)
        x_ans=SA(function=func,x=x_init,x_is_right=x_is_right)
        x_ans_avg+=x_ans/total_num
    print('x={:.2f} y={:.2f}'.format(x_ans_avg,func(x_ans_avg)))