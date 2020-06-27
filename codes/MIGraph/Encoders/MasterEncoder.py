

"""
Master class to vectorize node information.
see PrepDataset.py to understand the dependency of classes
"""

import numpy as np
import networkx as nx
import copy
from tqdm import tqdm
from multiprocessing import Pool


from Config import Config
from GraphUtil import obtainLabelList,is_num
from OtherEncoders import CategoryEncoder,ValueEncoder
from CompoundEncoder import CompEncoder
from WordEncoder import BERTEncoder
import time

CF=Config()
categoryEmbed=CF.categoryEmbed



#convert node information to vectors
class MasterEncoder:
    def __init__(self, graphProbList,CompDat,plot=True):
        """
        graphProbList: list of graphs
        CompDat: CompDatabase class

        """    
        #get compound, value, other node values as lists
        self.CList,self.VList,self.OList=obtainLabelList(graphProbList)

        #define encoders
        #self.OE=CategoryEncoder(self.OList,CF.ID_OTHERS)
        self.OE=BERTEncoder(CF.VALUE_DIM,CF.ID_OTHERS)
        
        #if initWordEncoder, calculate word vectors by BERT (and save them as dict) else just use saved vals.
        if CF.initWordEncoder:
            print("init BERT encoder")
            docList=list(set(self.OList))
            self.OE.initialize(docList)
        else:
            print("load BERT encoder")
            self.OE.load()
        
        self.CE=CompEncoder(CompDat,CF.ID_COMPOUNDS)
        self.VE=ValueEncoder(self.VList,CF.ID_VALUES)

        if plot:
            self.VE.plot()
            
    #get vector for a node value        
    def getVector(self,i):
        """
        i: node value
        return: its vector expression
        """
        if str(i).startswith("C_"):
            temp=i[2:]
            encoder=self.CE
        elif is_num(i):
            temp=i
            encoder=self.VE
        else:
            temp=i
            encoder=self.OE

        return encoder.getEVector(temp)
    
    #convert normal graphs to vectorized graphs (i.e, from normal text nodes to vector nodes)
    def convGraphToVecGraph(self, graph):
        """
        g: normal graph object
        return: graph object (vectorized)
        """
        g=copy.deepcopy(graph)
        
        #convert each nodes
        for nodeID in g.nodes:
            label=g.nodes[nodeID]["label"]
            ret=self.getVector(g.nodes[nodeID]["label"])
            
            if ret is None:
                return None
            
            g.nodes[nodeID]["label"]=ret
            
        return g
    
    #process multiple graphs
    def convGraphListToVecGraphList(self,graphList):
        print("converting graphs to vectors")
        vecGraphList=[]
        
        for graph in tqdm(graphList):
            ret=self.convGraphToVecGraph(graph)
            if ret is not None:
                vecGraphList.append(ret)
                
        return vecGraphList

    #parallel version
    def convGraphListToVecGraphList_parallel(self,graphList,split=2000):
        graphListList=zip(*[iter(graphList)]*split)
        totLength=(len(graphList)/split)
        
        with Pool(16) as p:
            result = list(tqdm(p.imap(self.convGraphListToVecGraphList, graphListList),total=totLength))
        
        res2=[]
        for i in result:
            res2.extend(i)

        return res2
    
    #convert graphs to adjacency matrixes and vectors    
    def convToMatrix(self,gList,vecMode=True):
        """
        gList: list of normal graphs
        vecMode: if false, unvectorized vals will be returned
        return: (list of adjacency matrixes, list of node vectors)
        
        """
        
        graphPropList=[]
        graphAdjList=[]
        
        #vectorize nodes for each node
        for g in tqdm(gList):
            for num,i in enumerate(g.nodes):

                if vecMode==False:
                    newVector=self.getVector(g.nodes[i]["label"])
                else:
                    newVector=g.nodes[i]["label"]
                    
                if num==0:
                    nodePropList=newVector
                else:
                    nodePropList=np.vstack((nodePropList,newVector))

            graphPropList.append(np.array(nodePropList,dtype="float32"))    

            #adjacency matrices
            graphAdjList.append(np.array(nx.adj_matrix(g).todense(),dtype="float32"))

        return graphPropList,graphAdjList

