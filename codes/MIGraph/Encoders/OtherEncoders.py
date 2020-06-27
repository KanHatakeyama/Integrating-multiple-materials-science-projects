"""
other encoders
"""

import numpy as np
import pandas as pd
import networkx as nx
import chainer.links as L
import matplotlib.pyplot as plt

from tqdm import tqdm
import joblib

from Config import Config

CF=Config()
categoryEmbed=CF.categoryEmbed


#value encoder
class ValueEncoder:
    def __init__(self,inputVList,num):
        #inputVList: list of values 
        #num: CF.ID_VALUES=2

        inputVList=np.array(inputVList).reshape(-1,1)
        self.num=num
        self.inputVList=inputVList
            
    def getVector(self,num):
        return (self.num,num)

    def getEVector(self,string):
        """
        string: inputted value (e.g, 1.2)
        return (category info, 1.2,1.2,1.2,....)
        """
        num,res=self.getVector(string)
        num=categoryEmbed(np.array([num])).array.reshape(-1)
        res=np.full(CF.VALUE_DIM, res).reshape(-1)
        return np.concatenate([num,res])
    
    def plot(self):
        x=[float(i) for i in self.inputVList]
        plt.hist(x)
        

#category encoder. this no longer necessary as long as bert encoder is used
class CategoryEncoder:
    def __init__(self,inputList,num):
        #inputlist:  list of words
        #num:  category ID
        
        self.le = preprocessing.LabelEncoder()
        self.le.fit(inputList)
        self.num=num
        self.valEmbed = L.EmbedID(len(self.le.classes_),CF.VALUE_DIM)
        

    def getVector(self,string):
        res=self.le.transform([str(string)])[0]
        return (self.num,res)
    
    def getEVector(self,string):
        num,res=self.getVector(string)
        res=self.valEmbed(np.array([res])).array
        num=categoryEmbed(np.array([num])).array
        return np.concatenate([num,res],1).reshape(-1)

