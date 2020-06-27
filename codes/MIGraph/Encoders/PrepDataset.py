
# coding: utf-8

"""
PrepDataset generates final data for machine learning


[dependency of classes]

-PrepDataset class (automatically calculate matrix and vectors from graph data)
    -Compound encoder class: (#TODO: this is duplicated with CE in master encoder)
    -MasterEncoder class
        - OE(other encoder): calc vectors for words
        - CE(compound encoder): calc vectors for compounds
        - VE(value encoder): calc vectors for values
        


"""


from Config import Config
from GraphUtil import convNodeToProblems,convAllNodeToProblems
from MasterEncoder import MasterEncoder
import warnings
from Fingerprint import Fingerprint
from CompDatabase import CompDatabase
from OtherEncoders import ValueEncoder
from tqdm import tqdm
import numpy as np

warnings.filterwarnings('ignore')
CF=Config()


class PrepDataset:
    def __init__(self,graphBasePath="graph/"):
        self.mode="Training"
        self.graphList=[]
        self.parallel=True
    
    #set compound encoder
    def setCompoundEncoder(self,compFilename):
        """
        compFilename: csv path for a file containing smiles
        calc descriptors and save those info as self.CompDat
        
        """
        #calc FPs
        FP=Fingerprint(CF.FINGERPRINT_DIM,4)
        FP.processCompFile(compFilename)
        self.FP=FP

        #set comp encoder
        CompDat=CompDatabase()
        CompDat.initialize(compFilename,FP.descDf,dimensionNum=CF.VALUE_DIM)
        
        self.CompDat=CompDat
    
    def initGraphList(self):
        self.graphList=[]
        
    #set master encoder    
    def setScaler(self):
        self.ALE=MasterEncoder(self.graphList,self.CompDat)
    
    #automatically convert to final dataset (i.e., make problems,etc )
    def convToDataset(self,genre=["C","V"],graphList=None,targetParams=CF.targetParams):
        """
        genre: target genre for preparing problems. if ["C","V"], compound and value nodes will be treated as problems.
                #TODO: this option is currently unavailable. only ["all"] is available.
        graphList: list of graphs for conversion
        targetParams: if this value is ["ionic conductivity"], only value nodes of ionic conductivity will be processed as problems
        """
        
        if graphList==None:
            graphList=self.graphList
            
        print("process ", len(graphList), " graphs")
        
        #convert node information to vectors
        if self.parallel:
            # parallel mode is faster with large datasets
            vecGraphList=self.ALE.convGraphListToVecGraphList_parallel(graphList)
        else:
            vecGraphList=self.ALE.convGraphListToVecGraphList(graphList)
            
        #make problems. (i.e., replace target parameter nodes with "__unknown__")
        if self.mode in ["Training","Test"]:
            
            #in this mode, all nodes can become "__unknown__"
            if genre ==["all"]:
                print("convert all nodes to problems")
                probVecGraphList,targetList,neighborNodeNameList=convAllNodeToProblems(graphList,vecGraphList,self.ALE)
            #normal mode: nodes with selected parameters(targetParams) will be used as problems
            else:
                probVecGraphList,targetList,neighborNodeNameList=convNodeToProblems(graphList,vecGraphList,self.ALE,genre,targetParams)
            
            targetList=np.array(targetList,"float32")

        else:
            #special mode. problems will not be made. (use this when e.g,, screening new compounds. users themselves must make "unknown" nodes manually)
            targetList=[np.array([0]*CF.CATEGORY_DIM,dtype=np.float32) for i in range(len(graphList))]
            probVecGraphList=vecGraphList
            neighborNodeNameList=[0]*len(targetList)
            
            
        #convert to adjacency matrix and node vectors
        adjList,propList=self.ALE.convToMatrix(probVecGraphList)

        #zip
        alldataset =[(i,j,k,l) for i,j,k,l in tqdm(zip(adjList,propList,targetList,neighborNodeNameList))]
        
        return alldataset
