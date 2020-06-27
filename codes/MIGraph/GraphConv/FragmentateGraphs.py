
"""
INCOMPLETED
fragmentate graphs, to increase the number of training data (, but not done in the paper.)
"""



import networkx as nx
from tqdm import tqdm
import copy
import joblib
import itertools
from joblib import Parallel, delayed
import random



def prepFragmentatedGList(gList,targetNode):
    """
    gList: list of graph
    targetNode: node id for a target node (i.e, please target a node ID of "C_****"). its neighboring nodes will be cut off
    e.g., [paramA]-[C_***]-[paramB] will be [C_****]-[paramA] and [C_****]-[paramB]
    """
    
    newGList=[]
    for g in (gList):
        neighbors=list(g.neighbors(targetNode))
        TFList = list(itertools.product([True,False],repeat=len(neighbors)))
        #print(len(TFList))
        
        # delete paramete nodes
        for removingFilter in TFList:
            removingNodeList = list(itertools.compress(neighbors, removingFilter))

            #delete neighbors
            if len(removingNodeList)<len(neighbors):
                ng=removeNodeGroups(g,targetNode,removingNodeList)
                newGList.append(ng)
            
    return newGList



#parallel mode...
def prepFragmentatedGList(gList,targetNode,numOfLists=20):
    """
    gList: list of graph
    targetNode: node id for a target node 
    numOfLists: maximum num of fragmentation
    """
    
    newGList=[]
    
    def conv(g,numOfLists=20):
        tempList=[]
        
        neighbors=list(g.neighbors(targetNode))
        arrayLen=len(neighbors)
        TFList=[[bool(random.getrandbits(1)) for i in range(arrayLen)] for i in range(numOfLists)]
        TFList=list(map(list, set(map(tuple, TFList))))
        
        
        # delete paramete nodes
        for removingFilter in TFList:
            removingNodeList = list(itertools.compress(neighbors, removingFilter))

            
            #del neighbors
            if len(removingNodeList)<len(neighbors):
                ng=removeNodeGroups(g,targetNode,removingNodeList)
                tempList.append(ng)
           
        return tempList
    
    print("originally ",len(gList),"graphs")
  
    newGListList = Parallel(n_jobs=-1,verbose=10)([delayed(conv)(n) for n in (gList)])
    
    
    #recover nested list
    for gList in newGListList:
        newGList.extend(gList)
    
    print("prepared ",len(newGList)," graphs")
    
    return newGList



#del unconnected nodes from the target node
def removeUnconnectedNodes(g,targetNode):
    removeList=[]
    
    for node in g.nodes:
        flg=nx.node_connectivity(g, s=targetNode, t=node)
        #print(flg,type(flg))
        if flg<1:
            removeList.append(node)
    
    for node in removeList:
        g.remove_node(node)            
        
#del target&neighboring nodes
def removeNodeGroups(g,keepingNode,removingNodeList):
    g=copy.deepcopy(g)    
    
    for removingNode in removingNodeList:
        g.remove_node(removingNode)    
        removeUnconnectedNodes(g,keepingNode)
    
    return g


"""
e.g.,

newGList=prepFragmentatedGList(gList,"compound")
drawGraph(newGList[16])

"""