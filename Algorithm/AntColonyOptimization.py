'''
假设我们需要寻找函数y=x1**2+x2**2+x3**3+x4**4在[1,30]之间的最大值
https://finthon.com/python-aco/
'''
import numpy as np
import random
import matplotlib.pyplot as plt

class ACO:
    def __init__(self,parameters):
        #para:种群大小popsize,迭代代数num_gen,自变量最值up,low,信息挥发系数rou,信息挥发总量Q,
        self.ngen=parameters['num_gen']
        self.popsize=parameters['popsize']
        self.num_var=len(parameters['up'])#变量个数
        self.bound=[]
        self.bound.append(parameters['low'])
        self.bound.append(parameters['up'])
        self.Q=parameters['Q']
        self.rou=parameters['rou']
        
        print('******parameters******')
        for key in parameters.keys():
            print("{} : {}".format(key,parameters[key]))
            
        self.pos_ant=np.zeros((self.popsize,self.num_var))#所有蚂蚁位置
        self.global_best=np.zeros((1,self.num_var))#全局最优位置
        
        #初始化蚁群
        tmp=-1
        for i in range(self.popsize):
            for j in range(self.num_var):
                self.pos_ant[i][j]=random.uniform(self.bound[0][j],self.bound[1][j])
            fit=self.function(self.pos_ant[i])
            if fit>tmp:
                tmp=fit
                self.global_best=self.pos_ant[i]
        
    def function(self,pos_ant):
        #根据问题修改函数(求解非线性多元函数最值)
        x1=pos_ant[0]
        x2=pos_ant[1]
        x3=pos_ant[2]
        x4=pos_ant[3]
        return (x1**2+x2**2+x3**3+x4**4)
    
    def prob_func(self,t,t_max):
        #对于非线性多元函数求极值
        #如果蚂蚁i的位置的信息素离信息素含量高的位置近，则进行微调
        p=[]
        for i in range(self.popsize):
            p.append((t_max-t[i])/t_max)
        return p
    
    '''
    def prob_tsp(self,t,eta,alpha,beta):
        for i in ran
    '''
    
    def update_operator(self,gen,t,t_max):#t为信息素,t_max为最大信息素
        lamda=1.0/self.ngen
        prob=self.prob_func(t,t_max)
        for i in range(self.popsize):
            for j in range(self.num_var):
                if prob[i]<random.uniform(0,1):#靠近最优，局部搜索
                    self.pos_ant[i][j]=self.pos_ant[i][j]+random.randint(-1,1)*lamda
                else:#远离最优，全局搜索
                    self.pos_ant[i][j]=self.pos_ant[i][j]+random.uniform(-1,1)*(self.bound[1][j]-self.bound[0][j])/2
                #越界保护
                if self.pos_ant[i][j]<self.bound[0][j]:
                    self.pos_ant[i][j]=self.bound[0][j]
                if self.pos_ant[i][j]>self.bound[1][j]:
                    self.pos_ant[i][j]=self.bound[1][j]
            #更新信息素值
            t[i]=(1-self.rou)*t[i]+self.Q*self.function(self.pos_ant[i])
            
            if self.function(self.pos_ant[i])>self.function(self.global_best):
                self.global_best=self.pos_ant[i]
        t_max=np.max(t)
        return t_max,t
    
    def main(self):
        popobj=[]
        best=np.zeros((1,self.num_var))[0]
        for gen in range(1,self.ngen+1):
            if gen==1:
                tmp=np.array(list(map(self.function,self.pos_ant)))
                t_max,t=self.update_operator(gen,tmp,np.max(tmp))
            else:
                t_max,t=self.update_operator(gen,t,t_max)
            popobj.append(self.function(self.global_best))
            if self.function(self.global_best)>self.function(best):
                best=self.global_best.copy()
            
            if gen%10==0:
                print("########## Generation {} ##########".format(gen))
                print('最优位置：{}'.format(self.global_best))
                print('最大函数值: {}'.format(self.function(self.global_best)))
        print('----------End of Algorithm----------')
        
        plt.figure()
        plt.title('Figure1')
        plt.xlabel('iterators',size=14)
        plt.ylabel('fitness',size=14)
        t=[t for t in range(1,self.ngen+1)]
        plt.plot(t,popobj,linewidth=2)
        plt.show()

if __name__=='__main__':
    para={
        'num_gen':1000,
        'popsize':100,
        'low':[1.0,1.0,1.0,1.0],
        'up':[30.0,30.0,30.0,30.0],
        'rou':0.8,
        'Q':1
    }
    aco=ACO(para)
    aco.main()
            
                    
                    

                