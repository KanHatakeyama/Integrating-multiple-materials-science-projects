"""
this is word encoder class
BertClient is needed for working

"""

import joblib
import re
import numpy as np
from bert_serving.client import BertClient
from sklearn.decomposition import PCA 


from Config import Config
CF=Config()
categoryEmbed=CF.categoryEmbed

class BERTEncoder:
    
    def __init__(self,VALUE_DIM,num,filePath="bin/BERT_docs.bin"):
        """
        VALUE_DIM: output dim
        num: category ID
        filePath: output file path to save BERT data.
        """
        self.VALUE_DIM=VALUE_DIM
        self.num=num
        self.filePath=filePath
        self.vecDict={}
        
    def initialize(self,textList):
        """
        textList: list of text in all graphs

        calculate their vectorized information by BERT and save them as dict
        """
        
        textList=[cleanText(txt) for txt in textList]
        with BertClient(port=5555, port_out=5556) as bc:
            textVecs = bc.encode(textList)
            
        #PCA
        pca = PCA(n_components=self.VALUE_DIM)
        embedding=pca.fit_transform(textVecs)
        
        #save as dict
        self.vecDict=dict(zip(textList,embedding))

        joblib.dump(self.vecDict,self.filePath)
                          
    def load(self):
        self.vecDict = joblib.load(self.filePath) 
        
    #get embedding vector
    def getEVector(self,string):
        """
        string: target word
        return: its vector
        """
        vec=self.getVector(cleanText(string))
        vec=vec.reshape(-1,self.VALUE_DIM)
        num=categoryEmbed(np.array([self.num])).array
        return np.concatenate([num,vec],1).reshape(-1)       

    #get original vector of the word
    def getVector(self,string):
        """
        string: target word
        return: its vector
        """
        vec=self.vecDict[string]
        return vec    
    
    
#utilities

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

cleanDict={
    "[":"",
    "]":"",
    "(":"",
    ")":"",
    "-":"",
}

def cleanText(txt):
    """
    txt: string
    return: cleaned text
    """

    for i in cleanDict:
        txt=txt.replace(i,cleanDict[i])

    txt=re.sub(r'[0-9]', "", txt)
    txt=re.sub(r'[︰-＠]', "", txt)
    txt=txt.lower()
    
    txt=txt.replace(" ","empty")
    
    if txt=="":
        txt="empty"
    
    return txt
        




