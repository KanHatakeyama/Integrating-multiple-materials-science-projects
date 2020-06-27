"""
utility functions about graphs
"""

import numpy as np
import pandas as pd
import networkx as nx
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

from Config import Config
from Funcs import is_num


CF=Config()
categoryEmbed=CF.categoryEmbed


def obtainLabelList(graphList):
    """
    input: list of graphs 
    return: list of labels on each node (compound label list, value list, other list)
    """

    #get all labels
    labelList=[]
    for g in graphList:
        for i in g.nodes:
            labelList.append(str(g.nodes[i]["label"]))

    CList=[]
    VList=[]
    OList=[]
    OList.append("_unknown_")

    #classification
    for i in labelList:
        #compound
        if i.startswith("C_"):
            CList.append(i[2:])
            continue
        #value
        elif is_num(i):
            VList.append(i)
            continue
        #others
        else:
            OList.append(i)

    return CList,VList,OList


#make problems from a graph
def convGraphToProblem(g,nodeID,encoder=None):
    """
    g: graph data
    nodeID: target node ID, which should be the answer of the problem. this node will be replaced with "_unknown_"
    encoder: if passed, "_unknown_" will be replaced with its embeddign vector (i.e., pass word encoder)
    return: (problem graph, answer val)
    """
    
    target=g.nodes[nodeID]["label"]
    g_new = copy.deepcopy(g)
    
    if encoder is None:
        label="_unknown_"
    else:
        label=encoder.getVector("_unknown_")
    
    g_new.nodes[nodeID]["label"]=label

    return g_new,target


#convert a series of graphs to problems
def convGraphListToProblems(graphList,targetUnit):
    """
    graphList: list of graphs
    targetUnit: target unit for problem preparation (e.g., "[S/cm]"). its corresponding value will be chosen as a problenm.
    return: (list of problem graphs, list of answer vals)
    """
    
    graphProblemList=[]
    targetList=[]

    for g in graphList:
        modifIDList=searchNodeIDforTargetUnit(g,targetUnit)

        for nodeID in modifIDList:
            ng,target=convGraphToProblem(g,nodeID)
            graphProblemList.append(ng)
            targetList.append(target)
            
    return graphProblemList,targetList


# make problems from graphs
# TODO: too complex...
def convNodeToProblems(graphList,vecGraphList,masterEncoder,genre=["C","V"],targetParams=CF.targetParams):
    """
    graphList:  list of graphs (before vectorization)
    vecGraphList: list of vectorized graphs (of graphList)
    masterEncoder: masterEncoder
    genre: genre to make problems: C: compound , V: numeric value, O: others  *** this mode may not work..?
    targetParams: target parametes to make problems
    return:  list of problem-type vectorized graphs, list of answers, list of parameter names of the answer
    """
    
    probVecGraphList=[]
    targetList=[]
    neighborNodeNameList=[]
    
    print("converting nodes to problems")

    for graphID,vecGraph in tqdm(enumerate(vecGraphList)):
        for node in vecGraph.nodes:
            nodeLabel=graphList[graphID].nodes[node]["label"]
            
            if str(nodeLabel).startswith("C_"):
                category="C"
            elif is_num(nodeLabel):
                category="V"
            else:
                category="O"

            #TODO: following codes are too complex and not clear....
            flg=False
            
            #find nodes of target parameters
            for neighborNode in graphList[graphID].neighbors(node):
                neighborNodeName=graphList[graphID].nodes[neighborNode]["label"]
                
                if CF.targetParamMode and neighborNodeName in targetParams:
                    flg=True
                    break

                #TODO: this func may not work with False..
                if CF.targetParamMode==False:    
                    flg=True

            #TODO: genre mode may not work
            if category in genre and flg ==True:
                g,target=convGraphToProblem(vecGraph,node,masterEncoder)
                probVecGraphList.append(g)
                neighborNodeNameList.append(neighborNodeName)
                
                if genre==["V"]:
                    targetList.append([target[-1]])
                else:
                    targetList.append(target[CF.CATEGORY_DIM:])
                
            
    return probVecGraphList,targetList,neighborNodeNameList


#convert all nodes to problems    
def convAllNodeToProblems(graphList,vecGraphList,masterEncoder):
    """
    vecGraphList: list of vectorized graphs
    masterEncoder: masterEncoder
    return: list of problems, list of answers
    """
    
    probVecGraphList=[]
    targetList=[]
    nodeNameList=[]
    
    for graphID,vecGraph in tqdm(enumerate(vecGraphList)):
        for node in vecGraph.nodes:
            g,target=convGraphToProblem(vecGraph,node,masterEncoder)
            probVecGraphList.append(g)
            targetList.append(target[CF.CATEGORY_DIM:])
            nodeNameList.append(graphList[graphID].nodes[node]["label"])
            
    return probVecGraphList,targetList,nodeNameList
