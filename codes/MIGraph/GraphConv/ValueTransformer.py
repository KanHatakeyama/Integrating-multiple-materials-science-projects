
"""
class which cleans and formats node contents
"""

import numpy as np
import pandas as pd
import networkx as nx
import copy
from tqdm import tqdm
import joblib
from Funcs import is_num
from ConvGraphScript import loadParamDictCSV,calcNodeValue,lowerGraph,searchNodeIDforTargetUnit


class ValueTransformer:
    def __init__(self):
        self.mode=None
        #get parameter setting dictionary
        self.logDict,self.unitDict=loadParamDictCSV()
        self.paramList=list(self.logDict.keys())
        
    def setMode(self,mode):
        self.mode=mode
    
    #format values
    def transform(self,val):
        """
        val: node value
        return: formatted (cleaned) node value
        """
        doLog=self.logDict[self.mode]
        unit=self.unitDict[self.mode]
        ret=calcNodeValue(val,doLog )
        
        #caution for invalid values (inf, nan)
        if ret==-np.inf or ret==np.inf or np.isnan(ret):
            print(val,"inf or nan!","  graph number: ",self.cnum, "  node: ",self.cnode, "  unit: ",self.cunit)
            
        return ret
    
    #clean graphs
    def convertGraphList(self,graphList):
        """
        graphList: list of graph objects
        return cleaned graph objects
        """
        
        tempList=copy.deepcopy(graphList)
        for num,graph in enumerate(tqdm(tempList)):
            self.cnum=num

            #clean values for each units
            for unit in self.paramList:
                self.setMode(unit)
                self.cunit=unit
                nodeList=searchNodeIDforTargetUnit(graph,unit)
                
                for node in nodeList:
                    self.cnode=node
                    temp=self.transform(graph.nodes[node]["label"])        
                    graph.nodes[node]["label"]=temp
                    
                    if temp>1500:
                        print("warning! very large value:",temp, "graph number:",num, "node: ",node)
        
            #every word will converted to small letters
            try:
                lowerGraph(graph)
            except:
                print("error convert. num: ",num)
            
        return tempList
    
