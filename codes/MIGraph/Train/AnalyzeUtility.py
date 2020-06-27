"""
utilities to analyze results
"""

import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda
from tqdm import tqdm
from GraphNNPredictor import formatDataset,myConcat
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
        
from Config import Config
CF=Config()


class AnalyzeUtility:
    def __init__(self,AutoSC,ggnn):
        self.AutoSC=AutoSC
        self.ggnn=ggnn

    def invSc(self,x,target="electric conductivity"):
        if target!="":
            return np.array([self.AutoSC.VSDict[target].inverse_transform(i) for i in x])
        else:
            return x

    def plot(self,x,y,sigma=None,target="electric conductivity"):
        if CF.genre==["V"]:
            if target!="":
                if sigma is not None:
                    sigma=self.invSc(y+sigma,target)-self.invSc(y,target)    
                x=self.invSc(x,target)
                y=self.invSc(y,target)
                    
        plt.figure(figsize=(5,5))
        plt.axes().set_aspect('equal', 'datalim')
        
        if sigma is not None:
            plt.errorbar(x, y, yerr = sigma, capsize=5, fmt='o', markersize=10, ecolor='blue', markeredgecolor = "blue", color='w',alpha=0.5)
            print("Sigma ave.: ",np.average(sigma))
        else:
            plt.plot(x,y,"o")
        
        print("R2: ",r2_score(x,y))
        print("MAE: ",mean_absolute_error(x,y))
        return x,y

    def predictByGGNN_batch(self,dataset,batchSize=32):
        splSize=batchSize
        
        tList=[]
        yList=[]
        
        for i in tqdm(range(int(len(dataset)/splSize)+1)):

            bgnIndex=i*splSize
            finIndex=(i+1)*splSize

            if len(dataset[bgnIndex:finIndex])==0:
                break
            
            cmp,adj,t=myConcat(dataset[bgnIndex:finIndex])
            y = self.ggnn(cmp,adj)
            
            y=cuda.to_cpu(y.data)
            t=cuda.to_cpu(t)
            
            tList.extend(t)
            yList.extend(y)

        return np.array(yList),np.array(tList)
        
def currentTime():
    return str(datetime.datetime.now())

