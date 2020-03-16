import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

tr=pd.read_csv("train.csv")

x=[]
for i in range(0,891):
    x.append((tr["Pclass"].as_matrix())[i])
    if (tr["Sex"].as_matrix())[i]=="male":
        x.append(1)
    if (tr["Sex"].as_matrix())[i]=="female":
        x.append(0)
    x.append(int((tr["Age"].as_matrix())[i])) if not np.isnan((tr["Age"].as_matrix())[i]) else x.append("29")

y=[]
for i in range(0,891):
    if (tr["Survived"].as_matrix())[i]==0:
        y.append(-1)
    else:
        y.append(+1)

xvec=np.reshape(x,(891,3))
xvec=xvec.astype(np.float)
yvec=np.reshape(y,(891,1))
yvec=yvec.astype(np.float)

