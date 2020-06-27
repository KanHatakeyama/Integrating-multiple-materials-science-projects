
"""
functions to convert graphML files and csv files, containing contents of onodes, to graph objects
"""

import numpy as np
import pandas as pd
import networkx as nx
import copy
from tqdm import tqdm
import joblib


#prepare graphs from graphml& csv files 
def loadGraphCSV(dataCSVFilename,graphBasePath):
    """
    dataCSVFilename: path to csv of node contents
    graphBasePath: path to a folder containing graphml files
    return: list of networkX objects
    
    """
    databaseDf=pd.read_csv(dataCSVFilename)

    #prepare dict of node parameters
    paramDictList=convDfToDict(databaseDf)

    #prepare graphs
    graphList=[]
    for i in tqdm(range(len(paramDictList))):
        g=obtainGraphFromDict(paramDictList[i],graphBasePath)
        graphList.append(g)
        
    return graphList


#convert dataframe (containing variables for graphs) information to dict
def convDfToDict(databaseDf):
    """
    input: dataframe
    output: dict
    """
    paramDictList=[]
    for dataID in range(databaseDf.shape[0]):
        paramDict={}
        paramDict["ID"]=databaseDf["ID"][dataID]
        paramDict["graph"]=databaseDf["graph"][dataID]

        #values
        valNames=databaseDf.iloc[dataID,4::2].values
        valval=databaseDf.iloc[dataID,5::2].values

        for i,j in zip(valNames,valval):
            if type(i)==type(""):
                paramDict[i]=j
        paramDictList.append(paramDict)

    return paramDictList


#prepare graph objects from variable information (of dict,) and graphs
def obtainGraphFromDict(paramDict,graphBasePath):
    """
    paramDict: dict of graph contents
    graphBasePath: path to graphML folder
    """
    
    #deceide graphML file to load
    graphFileName=graphBasePath+str(paramDict["graph"])+".graphml"

    #make a graph
    g=nx.read_graphml(graphFileName)
    g=g.to_undirected()

    #write node information from dict 
    for num,i in enumerate(g.nodes):
        nodeVal=g.nodes[i]["label"]
        if nodeVal in paramDict:
            g.nodes[i]["label"]=paramDict[nodeVal]
    return g




