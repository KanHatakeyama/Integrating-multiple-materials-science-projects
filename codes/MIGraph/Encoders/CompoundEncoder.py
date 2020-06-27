
"""
Compound encoder
this class returns vector information of a target smiles
"""

import numpy as np
from Config import Config

CF=Config()
categoryEmbed=CF.categoryEmbed


#compound encoder class
class CompEncoder:
    def __init__(self,CompDat,num):
        """
        CompDat: CompDatabase class
        num: =CF.ID_COMPOUNDS  (=3)
        """
        self.CompDat=CompDat     
        self.num=num
      
    #get embetting vector
    def getEVector(self,string):
        """
        input: SMILES
        return :embedding vector
        """
        num=self.num
        res=self.CompDat.getCompDesc(string)
        num=categoryEmbed(np.array([num])).array
        
        return np.concatenate([num,res],1).reshape(-1)
    