
"""
CompDatabase
this class converts structure information to n-dimensional vectors
"""


import numpy as np
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from Config import Config

CF=Config()
categoryEmbed=CF.categoryEmbed

    
    
#this class manages compound descriptor
class CompDatabase:
    def __init__(self):
        pass
    
    def initialize(self,f_comp,descDf,dimensionNum=16):
        """
        f_comp: path to compound database (with smiles )
        descDf: dataframe of fingerprint or desriptors of corresponding smiles list
        
        1) load smiles and original fingerprints(FPs)
        2) convert FPs to n-dim vectors by PCA etc (we call them "descriptors")
        3) manage them by dict (IDtoSMILES,IDtoDescs)
        """
        
        print("loading files")
        
        compDf=pd.read_csv(f_comp)
        descList=descDf.drop("SMILES",axis=1).columns

        mergeDf=pd.merge(compDf,descDf,on="SMILES")
        mergeDf=mergeDf.fillna(0)
        
        self.descList=descList
        self.IDList=mergeDf["ID"].values
        
        print("compressing and scaling")

        #PCA and standardizing 
        alldata=mergeDf[descList]
        self.AutoSC=AutoScale(dimensionNum=dimensionNum)
        self.compVecList=self.AutoSC.fit_transform(alldata)

        self.dimensionNum=dimensionNum
        
        #dict for ID and smiles
        self.IDtoSMILES = dict(zip(mergeDf["ID"].values,mergeDf["SMILES"].values))

        #dict fo id and properties
        self.IDtoDescs = dict(zip(mergeDf["ID"].values,self.compVecList))

        
    #get compound descriptors for a specific compound 
    def getCompDesc(self,compID):
        """
        compID: compound ID
        return: its descriptor
        """
        
        try:
            ret=self.IDtoDescs[compID].reshape(1,-1)
        except:
            #if not found, return zeros
            print("unknown compound: ",compID)
            ret=np.zeros(self.dimensionNum).reshape(-1,self.dimensionNum)
            
        return ret

    def inverse_transform(self,vec):
        """
        vec: descriptor
        return: FP after decompressing
        """
        return self.AutoSC.inverse_transform(vec)



    #search compounds having similar descriptors
    def searchCompFromCompVec(self,targetvec,num=100):
        """
        targetvec: descriptor to be compared
        num: maximum number of returns
        return: similar compounds found in this class (as a dataframe)
        """
        
        compVecs=self.compVecList
        
        #calculate cos similarity
        cosSimList=np.dot(compVecs,targetvec)/(np.linalg.norm(targetvec)* (np.linalg.norm(compVecs,axis=1)))
        distList=cosSimList
        
        try:
            self.mergeD
        except:
            mergeDf=pd.DataFrame()
            mergeDf["ID"]=self.IDtoSMILES.keys()
            mergeDf["SMILES"]=self.IDtoSMILES.values()
            self.mergeDf=mergeDf

        resDf=self.mergeDf[["ID","SMILES"]]
        resDf["Cos_sim"]=list(distList)
        resDf=resDf.sort_values("Cos_sim",ascending=False)
        
        
        return resDf[:num]


    
#automatic scaling class for compounds
#it does standardizing, PCA, and standardizing
class AutoScale:
    def __init__(self,dimensionNum):
        self.dimensionNum=dimensionNum

        self.sc1=StandardScaler()
        self.pca = PCA(n_components=dimensionNum)
        self.sc2=StandardScaler()
    
        self.scList=[self.sc1,self.pca,self.sc2]        


    def fit_transform(self,alldata):        
        for scaler in self.scList:
            alldata=scaler.fit_transform(alldata)
        return alldata

    def transform(self,X):
        for scaler in self.scList:
            X=scaler.transform(X)
        return X

    def inverse_transform(self,X):
        self.scList.reverse()
        for scaler in self.scList:
            X=scaler.inverse_transform(X)
            
        self.scList.reverse()
        
        return X

