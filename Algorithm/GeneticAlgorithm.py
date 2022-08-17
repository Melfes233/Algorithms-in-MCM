'''
例：寻找函数y=x1**2+x2**2+x3**3+x4**4在[1,30]之间的最大值
https://finthon.com/python-genetic-algorithm/
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from operator import itemgetter

class Gene:
    def __init__(self,**data):
        self.__dict__.update(data)
        self.size=len(data['data'])
        
class GA:
    def __init__(self,parameter):
        #parameter=[crossprob,mutationprob,popsize,low,up]交叉率，变异率，种群大小，自变量可取的最小值与最大值
        self.parameter=parameter
        low=self.parameter['low']
        up=self.parameter['up']
        self.bound=[]
        self.bound.append(low)
        self.bound.append(up)
        self.popsize=self.parameter['popsize']
        print('******parameters******')
        for key in parameter.keys():
            print("{} : {}".format(key,parameter[key]))
        
        pop=[]
        for i in range(self.parameter['popsize']):
            geneinfo=[]
            for j in range(len(low)):
                geneinfo.append(random.uniform(self.bound[0][j],self.bound[1][j]))
            #评估基因适应性
            fitness=self.evaluate(geneinfo)
            #加入种群
            pop.append({'Gene':Gene(data=geneinfo),'fitness':fitness})
        
        self.pop=pop
        #选出最优个体
        self.bestone=self.selectBest(self.pop)
    
    def evaluate(self,geneinfo):
        #计算适应性
        #根据实际问题修改
        x1=geneinfo[0]
        x2=geneinfo[1]
        x3=geneinfo[2]
        x4=geneinfo[3]
        return (x1**2+x2**2+x3**3+x4**4)
    
    def selectBest(self,pop):
        #选出最好的个体
        sorted_list=sorted(pop,key=itemgetter('fitness'),reverse=True)
        return sorted_list[0]
    
    def selection(self,individuals,k):
        #按照概率从上一代种群中选择k个个体,(选完K个后)形成新的一代
        sorted_list=sorted(individuals,key=itemgetter('fitness'),reverse=True)
        
        sum_fits=sum(i['fitness'] for i in individuals)
        
        chosen=[]
        #适应性高的更容易被选到
        for i in range(k):
            chosen_p=random.uniform(0,1)*sum_fits
            sum_p=0#累积概率
            for ind in sorted_list:
                sum_p+=ind['fitness']
                if sum_p>chosen_p:
                    chosen.append(ind)
                    break
        chosen=sorted(chosen,key=itemgetter('fitness'),reverse=True)
        return chosen

    
    def crossoperate(self,offspring,used_type=0):
        #两个个体的基因片段随机进行单点交叉或双点交叉
        dim=len(offspring[0]['Gene'].data)
        
        geneinfo1=offspring[0]['Gene'].data
        geneinfo2=offspring[1]['Gene'].data
        
        if used_type==1 or used_type==2:
            cross_type=used_type
        else:
            cross_type=random.randint(1,2)
        
        if dim==1:#基因长度为1,不进行交换
            pos1=1
            pos2=1
        else:
            pos1=random.randint(0,dim-1)
            pos2=random.randint(0,dim-1)
        
        newoff1=Gene(data=[])
        newoff2=Gene(data=[])
        tmp1=[]
        tmp2=[]
        if cross_type==2:
            for i in range(dim):
                if min(pos1,pos2)<=i<max(pos1,pos2):
                    tmp1.append(geneinfo1[i])
                    tmp2.append(geneinfo2[i])
                else:
                    tmp1.append(geneinfo2[i])
                    tmp2.append(geneinfo1[i])
        else:
            for i in range(dim):
                if i<=pos1:
                    tmp1.append(geneinfo1[i])
                    tmp2.append(geneinfo2[i])
                else:
                    tmp1.append(geneinfo2[i])
                    tmp2.append(geneinfo1[i])
                    
        newoff1.data=tmp1
        newoff2.data=tmp2
        return newoff1,newoff2
    
    def mutation(self,crossoff,bound,used_type=0):
        #单点基本位变异与逆转变异
        
        dim=len(crossoff['Gene'].data)
        newoff=crossoff['Gene']
        
        if dim==1:
            pos=0
        else:
            pos=random.randint(0,dim-1)
            
        if used_type==1 or used_type==2:
            mut_type=used_type
        else:
            mut_type=random.randint(1,2)
            
        if mut_type==1:
            newoff.data[pos]=random.randint(bound[0][pos],bound[1][pos])
        else:
            revpos=random.randint(0,dim-1)
            tmp=newoff.data[pos]
            newoff.data[pos]=newoff.data[revpos]
            newoff.data[revpos]=tmp
            
        return newoff
    
    def GA_main(self):
        popsize=self.parameter['popsize']
        print("Start evolution")
        
        
        for g in range(self.parameter['num_generation']):
            selectpop=self.selection(self.pop,self.popsize)
            nextoff=[]
            while len(nextoff)<self.popsize:
                #选出最优的前两个进行遗传操作
                tmp=selectpop.pop()
                offspring=[tmp]
                resoff=[]
                
                if len(selectpop)>1 and random.uniform(0,1)<self.parameter['crossprob']:#交叉操作
                    offspring.append(selectpop.pop())
                    off1,off2=self.crossoperate(offspring)
                    resoff+=[off1,off2]
                    
                if random.uniform(0,1)<self.parameter['mutationprob']:
                    for i in range(len(offspring)):
                        off=self.mutation(offspring[i],self.bound,used_type=1)
                        resoff.append(off)
                    
                for i in range(len(resoff)):
                    fit_score=self.evaluate(resoff[i].data)
                    nextoff.append({'Gene':resoff[i],'fitness':fit_score})
                    
                nextoff.extend(offspring)
                
            self.pop=nextoff
            
            fits=[ind['fitness'] for ind in self.pop]
            
            best_ind=self.selectBest(self.pop)
            
            if best_ind['fitness']>self.bestone['fitness']:
                self.bestone=best_ind
            
            if (g+1)%500==0:
                print("##########Generation {}##########".format(g+1))
                print("Best individual found is {}, {}".format(self.bestone['Gene'].data,self.bestone['fitness']))
                print("Max fitness of current pop: {}".format(max(fits)))
    
        print("-----End of evolution-----")

if __name__=='__main__':
    para={'crossprob':0.5,
          'mutationprob':0.5,
          'num_generation':10000,
          'popsize':100}
    up=[30,30,30,30]
    low=[1,1,1,1]
    para['up']=up
    para['low']=low
    new_ga=GA(parameter=para)
    new_ga.GA_main()
        