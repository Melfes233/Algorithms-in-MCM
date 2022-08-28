import sys
import os
import pandas as pd
import json
sys.path.append(os.path.dirname(os.path.abspath('__file__')))


from Analysis.AHP import *
from Evaluation.TOPSIS import *
import numpy as np

def read_xl(input_path):
    workbook=pd.read_excel(input_path,usecols=[1,2,3,4,5])
    names=pd.read_excel(input_path,usecols=[0])
    data=workbook.values
    return data,names.values

def TOPSIS_2021C_main():
    a=np.array([[1  ,6  ,8  ,5  ,4  ],
                [1/6,1  ,3  ,1  ,1/4],
                [1/8,1/3,1  ,1/5,1/5],
                [1/5,1  ,5  ,1  ,1/3],
                [1/4,4  ,5  ,3  ,1  ]
    ])
    w=AHP(a,5)
    w=w.reshape(5)
    print('weight:',w)
    print('-----------------------------------------')
    input_file=os.path.join(sys.path[0],'data','data.xlsx')
    data,names=read_xl(input_file)
    print(names[0],data[0])
    data_type=[0,0,0,0,3]
    low=0.95
    up=1.05
    score=TOPSIS_main(data,weight=w,data_type=data_type,low=low,up=up)
    # score=[round(k,10) for k in score]
    res=[{'name':names[i][0],'score':score[i]} for i in range(len(names))]
    
    def getelem(elem):
        return elem['score']
    
    res.sort(key=getelem,reverse=True)

    with open(os.path.join(sys.path[0],'data','scores.json'),'w',encoding='utf-8') as f:
        for i in res:
            json.dump(i,f,ensure_ascii=False)
            f.write('\n')

if __name__=='__main__':
    TOPSIS_2021C_main()
   