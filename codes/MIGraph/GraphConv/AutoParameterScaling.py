"""
classes which standardize values in graphs automatically
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import glob
from sklearn.preprocessing import StandardScaler
from ConvGraphScript import searchNodeIDforTargetUnit


#Wrapper of standardscaler
class ValueStandardizer:
    def __init__(self,paramName):
        self.paramName=paramName
        self.valList=[]
        
    #add value to the class
    def add(self,val):
        self.valList.extend(val)

    def fit(self):
        sc = StandardScaler()
        valList=np.array(self.valList).reshape(-1,1)
        sc.fit(valList)
        self.sc=sc
    
    def transform(self,val):
        """
        input: float
        return float
        *inputs and returns should not be array
        """
        if val=="_unknown_":
            return val
        else:
            val=float(val)
        
        val=np.array(val).reshape(-1,1)
        return self.sc.transform(val)[0][0]
    
    def inverse_transform(self,val):
        val=np.array(val).reshape(-1,1)
        return self.sc.inverse_transform(val)[0][0]

    
#automatically standardize values in graph objects (for each units)
class AutoParameterScaling:
    def __init__(self,unitListPath="UnitList.csv"):
        #load setting dictionary
        unitDf=pd.read_csv(unitListPath)
        paramNames=unitDf["Name"]
        paramNames=[str(i).lower() for i in paramNames]
        
        #prepare ValueStandardizer classes for each unit
        VSDict={}
        for num,name in enumerate(paramNames):
            VS=ValueStandardizer(name)
            VSDict[name]=VS
        
        self.paramNames=paramNames
        self.VSDict=VSDict

    #preparing standardizers
    def initialize(self,graphbinList):
        """
        input: path lists of graph objects
        """
        paramNames=self.paramNames
        VSDict=self.VSDict
        print("loading files")
        
        #search values in graphs and record them to standardizers
        for path in tqdm(graphbinList):
            gList=joblib.load(path)    
            for g in (gList):
                for num,name in enumerate(paramNames):
                    nodeIDList=(searchNodeIDforTargetUnit(g,name))
                    valList=[float(g.nodes[node]["label"]) for node in nodeIDList]    
                    VSDict[name].add(valList)

        #fit
        print("fitting scalers")
        for name in (paramNames):
            try:
                VSDict[name].fit()
            except:
                print("fit failed: ",name)
      
        self.VSDict=VSDict
    
    #automatically standardize values in every graph
    def autoTransform(self,graphbinList):
        """
        input: path lists of graph objects
        """
        paramNames=self.paramNames
        VSDict=self.VSDict
        
        #standardizing
        for path in tqdm(graphbinList):
            gList=joblib.load(path)    
            for g in (gList):
                for name in (paramNames):
                    nodeIDList=(searchNodeIDforTargetUnit(g,name))
                    valList=[(g.nodes[node]["label"]) for node in nodeIDList]    

                    #replace values with standarized ones
                    scaledValList=[VSDict[name].transform(i) for i in valList]
                    for num,nodeId in enumerate(nodeIDList):
                        g.nodes[nodeId]["label"]=scaledValList[num]


            #saving
            #TODO: file path should be able to be changed by users
            newPath=path.replace("../convCSVtoGraph/temp/output/","output/")
            newPath=newPath.replace("temporary/","output/")
            print("saving: ",newPath)
            joblib.dump(gList,newPath)