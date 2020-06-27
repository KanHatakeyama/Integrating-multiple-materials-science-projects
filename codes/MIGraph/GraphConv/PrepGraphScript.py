
"""
utility functions to prepare graph objects
"""

"""
class to prepare graphs from "script-type" csv files
#TODO function names are rather confusing...
"""

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import joblib
import os
from Funcs import is_num
from FragmentateGraphs import prepFragmentatedGList
from ConvGraphScript import loadParamDictCSV,addParams,lowerGraph,checkGraphList


class PrepGraphScript:
    def __init__(self,CSVFilename):
        """
        CSVFilename: path to "script-type" csv
        """
        self.df=pd.read_csv(CSVFilename)
        
        #dict of parameter setting
        self.logDict,self.unitDict=loadParamDictCSV()
        
        CSVFilename = os.path.basename(CSVFilename)
        self.filename=CSVFilename.replace(".csv","")
        
        #private varuables
        self.processID=0  #index in csv
        self.gList=[]     
        self.g=None      
 
        #fragmentation mode
        self.doFragment=True

    def initGraph(self):
        self.g = nx.Graph()
    
    #set node values
    def setNode(self,columnName,nodeID="cmp",isCompound=True,isSMILES=True):
        """
        columnName: target columun name in DF
        nodeID: target node IF
        isCompound: if compound or not. if so, node label will become "C_****"
        return: list of nodets
        
        """
        
        nodeLabel=self.df[columnName][self.processID]
        nodeLabel=str(nodeLabel)

        if isCompound:
            #one compound
            if nodeLabel.find("/")<0 or isSMILES==True:
                nodeLabel="C_"+nodeLabel
                self.g.add_node(nodeID,label=nodeLabel)   
                return [nodeID]
                
            #composition of compounds
            else:
                nodeLabelList=nodeLabel.split("/")
                IDList=[]
                for num,i in enumerate(nodeLabelList):
                    nodeLabel="C_"+i
                    temp=nodeID+"_"+str(num)
                    self.g.add_node(temp,label=nodeLabel)   
                    IDList.append(temp)
                return IDList
            
        else:
        #others
            self.g.add_node(nodeID,label=nodeLabel)        
            return [nodeID]
        
    #get parameter list to be added from CSV
    def autoObtainParameterListFromCSV(self,verbose=False):
        """
        return: parameters to be added
        
        """
        
        targetParamns=list(self.df.columns & self.unitDict.keys())
        if verbose:
            print("Process the following paramns: ", targetParamns)
            print("\nFollowing columns were ignored: ", self.df.columns ^ targetParamns)
        return  targetParamns
    
    #write target parameter information to target node (i.e., [compound] to  [compound]-[value]-[unit])
    def addParamToNode(self, targetNode,targetParam):
        """
        targetNode: target node ID
        targetParam: array of target parameter information
        """
        
        #from dict, check target parameter's unit and necessity of log conversion
        paramUnit=self.unitDict[targetParam]
        paramLog=self.logDict[targetParam]
        val=self.df.iloc[self.processID,:][targetParam]
        
        #in csv, multiple values may be recorded with slash (eg, 123/23)
        valList=str(val).split("/")
        
        #add value and unit nodes
        for num,i in enumerate(valList):
            params=(targetParam+str(num),targetParam,i,paramLog,paramUnit)
            self.g=addParams(self.g,params,targetNodeID=targetNode)
   
        
    #make graphs from this function can be modeified by override in the cases of special "script-type" csvs
    #as a default, this function makes graphs from csv with the columns of "SMILES - parammeter name 1 - value 1 - parameter name 2 - value2 -..."
    def convertGraph(self):
        self.initGraph()
        self.setNode("SMILES",nodeID="compound")
        addParamList=self.autoObtainParameterListFromCSV()
        for param in addParamList:
            self.addParamToNode("compound",param)
            
        #save smiles list
        #TODO: currently, file path is fixed
        smDf=pd.DataFrame()
        smDf["SMILES"]=self.df["SMILES"]
        smDf["ID"]=+smDf["SMILES"]
        smilesPATH="input/"+self.filename+".csv.gz"
        smDf[["ID","SMILES"]].to_csv(smilesPATH,index=False)
            
            
    #automatically prepare graphs
    def prapareGraphList(self,numOfMaxFragments=20):
        for i in tqdm(range(self.df.shape[0])):
            self.processID=i
            self.convertGraph()
            lowerGraph(self.g)
            self.gList.append(self.g)
            
        checkGraphList(self.gList)
        
        #graphs can be fragmentated to increase the number of training data (but not conducted in the paper)
        if self.doFragment:
            print("fragmentating...")
            self.gList=prepFragmentatedGList(self.gList,"compound",numOfLists=numOfMaxFragments)
        
        #TODO; currently, file path is fixed
        outname="temporary/"+self.filename+".graphbin"
        print("saving...", outname)
        joblib.dump(self.gList,outname,compress=3)
    