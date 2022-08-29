import sys
import os
import pandas as pd
import json
import xlwt
sys.path.append(os.path.dirname(os.path.abspath('__file__')))


from Analysis.AHP import *
from Evaluation.TOPSIS import *
import numpy as np

def read_xl(input_path):
    workbook=pd.read_excel(input_path,sheet_name='数据总表',usecols=[1,2,3,4,5])
    names=pd.read_excel(input_path,usecols=[0])
    data=workbook.values
    return data,names.values

def read_xl2(input_path):
    workbook=pd.read_excel(input_path,sheet_name='Sheet2',usecols=[1,2])
    names=pd.read_excel(input_path,sheet_name='Sheet2',usecols=[0])
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
    res=[{'name':names[i][0],'score':score[i]} for i in range(402)]
    
    def getelem(elem):
        return elem['score']
    
    res.sort(key=getelem,reverse=True)


    book=xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet=book.add_sheet('score',cell_overwrite_ok=True)
    col=['供货商','得分']
    for i in range(len(col)):
        sheet.write(0,i,col[i])
    for i in range(402):
        sheet.write(i+1,0,res[i]['name'])
        sheet.write(i+1,1,res[i]['score'])
    save_path=os.path.join(sys.path[0],'data','score.xls')
    book.save(save_path)

def TOPSIS_2021C_main2():
    input_file=os.path.join(sys.path[0],'data','T_data.xlsx')
    data,names=read_xl2(input_file)
    
    data_type=[1,1]
    w=np.array([0.8,0.2])
    print(data[0])
    score=TOPSIS_main(data,weight=w,data_type=data_type)
    # score=[round(k,10) for k in score]
    res=[{'name':names[i][0],'score':score[i]} for i in range(8)]
    
    def getelem(elem):
        return elem['score']
    
    res.sort(key=getelem,reverse=True)

    book=xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet=book.add_sheet('score',cell_overwrite_ok=True)
    col=['转运商','得分']
    for i in range(len(col)):
        sheet.write(0,i,col[i])
    for i in range(8):
        sheet.write(i+1,0,res[i]['name'])
        sheet.write(i+1,1,res[i]['score'])
    save_path=os.path.join(sys.path[0],'data','T_score.xls')
    if os.path.exists(save_path):
        os.remove(save_path)
    book.save(save_path)


if __name__=='__main__':
    TOPSIS_2021C_main2()
   