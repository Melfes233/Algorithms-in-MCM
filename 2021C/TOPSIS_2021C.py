import sys
import os
import openpyxl
sys.path.append(os.path.dirname(os.path.abspath('__file__')))


from Analysis.AHP import *
import numpy as np

def read_xl():

if __name__=='__main__':
    a=np.array([[1  ,6  ,8  ,5  ,4  ],
                [1/6,1  ,3  ,1  ,1/4],
                [1/8,1/3,1  ,1/5,1/5],
                [1/5,1  ,5  ,1  ,1/3],
                [1/4,4  ,5  ,3  ,1  ]
    ])
    w=AHP(a,5)
    w=w.reshape(5)

   