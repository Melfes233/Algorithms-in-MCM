import openpyxl
import os
import sys
import numpy as np

def load(input_path):
    f=openpyxl.load_workbook(input_path)
    sheets=f.sheetnames
    # print(sheets)
    worksheet=f[sheets[0]]
    # print(worksheet)
    i=0
    row_title=[]
    col_title=[]
    type_title=[]
    data=np.ones([402,240])
    for row in worksheet.rows:
        i=i+1
        if i==1:
            for cell in row:
                row_title.append(cell.value)
            continue
        elif i-2>=402:
            break
        j=0
        for cell in row:
            j=j+1
            if j==1:
                col_title.append(cell.value)
            elif j==2:
                type_title.append(cell.value)
            elif j-3>=240:
                break
            else:
                if cell.value==None:
                    data[i-2,j-3]=-1
                else:
                    data[i-2,j-3]=cell.value
    # print(row_title)
    # print(col_title)
    # print(type_title)
    return data,col_title

def calculate(data,u,v):
    res=np.ones(data.shape[0])
    for i in range(data.shape[0]):
        num=0
        total=0
        for j in range(data.shape[1]):
            if data[i,j]==-1:
                continue
            num=num+1.0
            if data[i,j]>=1:
                tmp=u*(data[i,j]-1.0)
            else:
                tmp=v*(1.0-data[i,j])
            total=total+tmp
        res[i]=total/num
    return res
            
        
if __name__=='__main__':
    input_file=os.path.join(sys.path[0],'data','supply1.xlsx')
    data,col_title=load(input_file)
    res=calculate(data,1,1)

    with open(os.path.join(sys.path[0],'data','supply.txt'),'w',encoding='utf-8') as f:
        for i in range(len(res)):
            # f.write('{}:\t\t{}\n'.format(col_title[i],res[i]))
            f.write('{}\n'.format(res[i]))