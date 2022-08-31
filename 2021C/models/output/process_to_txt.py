import json
import os
import sys


if __name__=='__main__':
    path=os.path.join(sys.path[0],'chosen')
    names=os.listdir(path)
    for name in names:
        if name[0]!='S':
            continue
            
        input_path=os.path.join(path,name)
        output_path=os.path.join(path,'pretxt',name[:-4]+'txt')
        if os.path.exists(output_path):
            os.remove(output_path)
        pre=[]
        with open(input_path,'r',encoding='utf-8') as f:
            data=json.loads(f.readline())
            pred=data['prediction']
            for i in range(len(pred)):
                pre.append(pred[i]['pred'])
        
        with open(output_path,'w',encoding='utf-8') as f:
            for i in pre:
                f.write(str(i))
                f.write('\n')
