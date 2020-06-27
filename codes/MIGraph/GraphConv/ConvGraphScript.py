
"""
utilitie about graph objects
"""
import pandas as pd
import numpy as np
import networkx as nx
import re
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from Funcs import is_num
from requests.structures import CaseInsensitiveDict
from FragmentateGraphs import prepFragmentatedGList


#draw graph
def drawGraph(g,drawLabel=True,printNodes=False):
    """
    g: networkX object
    drawLabel: draw labels if True else node ID
    """
    pos = nx.spring_layout(g, k=0.5)
    plt.figure(3,figsize=(12,12)) 

    if drawLabel:
        gDict={}
        for node in g.nodes:
            gDict[node]=g.nodes[node]["label"]

        nx.draw(g,pos, with_labels = True,labels=gDict)
    else:
        nx.draw(g,pos, with_labels = True)

    if printNodes:
         for node in g.nodes:
            print(node,"  ",g.nodes[node]["label"])       

#search nodes having a specific node value
def searchTxtFromGraph(g,txt):
    """
    g: graph objext
    txt: node value to be searched
    """
    nodeVals=[g.nodes[node]["label"] for node in g.nodes]
    return txt in nodeVals

#search graphs having a specific node value
def searchTxtFromGraphList(gList,txt):
    """
    g: list of graph objext
    txt: node value to be searched
    """    
    TFList=[searchTxtFromGraph(g,txt) for g in gList]
    gDict=dict(zip(gList,TFList))  
    gList = [k for k, v in gDict.items() if v == True]
    return  gList
            
#convert node values by some rules
def calcNodeValue(val,doLog):
    """
    val: node value
    doLog: if true, convert to log scale (but slightly modified, see below)
    return processed value
    """
    #do nothing in the case of "unknown"
    if val=="_unknown_":
        return val
    
    if doLog:
        logval=np.log10(float(val)+1/10**4)
    else:
        if is_num(val):
            logval=float(val)
        else:
            logval=val
            
    return logval
            
#add parameter information to the node
def addParams(g,params,targetNodeID="comp"):
    """
    g: graph object
    prams: array of (node name, label name, value to be written, log scale or not, unit name)
    targetNodeID: target node id
    return graph
    """
    nodeName,labelName,val,doLog,unit=params[0],params[1],params[2],params[3],params[4]

    if is_num(val) or ((val is not np.nan and val==val and val is not None) and val!="unknown" and val!="nan"):
        logval=calcNodeValue(val,doLog)
        g.add_node(nodeName,label=labelName)
        g.add_edge(targetNodeID,nodeName)

        g.add_node(nodeName+"_val",label=str(logval))
        g.add_edge(nodeName+"_val",nodeName)

        if unit!="[No unit]":
            g.add_node(nodeName+"_unit",label=unit)
            g.add_edge(nodeName+"_unit",nodeName+"_val")
    return g

#load dictionary of parameter setting
def loadParamDictCSV(configPATH="UnitList.csv"):
    """
    return: logDict: if the parameter will be converted to log scale
    return: unitDict: unit name
    """
    configDf=pd.read_csv(configPATH)
    configDf["log"]=[configDf["log"]=="+"][0]
    
    logDict=CaseInsensitiveDict(configDf[["Name","log"]].values)
    unitDict=CaseInsensitiveDict(configDf[["Name","Unit"]].values)
    
    return logDict,unitDict


#check graph contents
def checkGraphList(gList,valThreshold=1500,showGraph=False):
    """
    gList: list of graph
    valThreshold: absolute threshold of checking
    if unfamiliar values are detected, it will be printed
    """
    print("checking graph values...")
    
    for num,g in enumerate(gList):
        for node in g.nodes:
            val= (g.nodes[node]["label"])

            if is_num(val):
                absVal=(np.abs(float(val)))

                if absVal>valThreshold:
                    print("caution: too large val: ",val, "graph No.: ",num)
                    if showGraph:
                        drawGraph(g,printNodes=True)
    print("check done")
    
    
#convert every word to small letters (except for compound informationm, "C_****")
def lowerGraph(g):
    """
    g: graph
    """
    for node in g.nodes:
        labelName=g.nodes[node]["label"]
        
        if str(labelName)[:2]!="C_" and is_num(labelName)==False:
            try:
                g.nodes[node]["label"]=labelName.lower()
            except:
                print("lowering error with label:",labelName)

#get node id for a target parameter 
def searchNodeIDforTargetUnit(g,targetLabel):
    
    """
    g: graph
    targetLabel: target label to be searched (e.g., Dipole Moment)
    return: list of nodeID
    
    e.g, if the following graph was searched, node id of "23" will be returned
    [C_111]-[Dipole moment]-[23]-[-]
    """
    
    neighborNodeList=[]
    for num,i in enumerate(g.nodes):

        currentLabel=g.nodes[i]["label"]
        if targetLabel ==currentLabel:

            #check neighbor nodes
            for neighborNode in g.neighbors(i):
                #some of neightbor nodes should be values
                if is_num(g.nodes[neighborNode]["label"]):
                    neighborNodeList.append(neighborNode)
    return neighborNodeList