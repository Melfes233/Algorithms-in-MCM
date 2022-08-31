import json
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

if __name__=='__main__':
    path=os.path.join(sys.path[0],'models','output','chosen')
    names=os.listdir(path)
    workbook=pd.read_excel(os.path.join(sys.path[0],'data','data_chosen.xlsx'),usecols=range(1,242))
    wdata=workbook.values
    for name in names:
        if name[0]!='S':
            continue
        input_path=os.path.join(path,name)
        name=name[:-5]   
        pre=[]
        with open(input_path,'r',encoding='utf-8') as f:
            data=json.loads(f.readline())
            pred=data['prediction']
            for i in range(len(pred)):
                pre.append(pred[i]['pred'])

        # print(data)
        for i in range(wdata.shape[0]):
            if str(wdata[i][0])==name:
                pre=wdata[i][1:].tolist()+pre
        
        plt.plot(range(1,289),pre)
        plt.title(name)
        plt.show()

    