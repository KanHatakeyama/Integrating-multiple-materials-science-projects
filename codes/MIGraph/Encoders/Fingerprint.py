
"""
Fingerprint class
this class calculates fingerprint (FP) (or its vectors) of a smiles

1) read list of smiles from csv
2) calc FP (0001|0011|...)
3) convert FP to vectors (1,3,...)

#TODO there may be a better way to vectorize chem info than this way(e.g., molecular descriptors, neural net, etc.).
advantage of this way is fastness!

"""

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm

class Fingerprint:
    def __init__(self, BIT_LEN=2048,BIT_DIV=4):
        """
        BIT_LEN: bit length of FP
        BIT_DIV: (BIT_LEN/BIT_DIV)-dimensional vectors will be made
                 e.g.:  FP of (0001|0000|...) will be (1,0,0,...) 
        """
        self.BIT_LEN=BIT_LEN
        self.BIT_DIV=BIT_DIV
        
    def LoadSMILESList(self,compFilename):
        """
        compFilename: path to the compound csv file
        """
        self.smilesList=getSMILESList(compFilename)
        
    def calcFingerprintBit(self,smiles):
        """
        smiles: SMILES string
        return: FP
        """
        mol = Chem.MolFromSmiles(smiles)
        morgan_fps =AllChem.GetMorganFingerprintAsBitVect(mol, 2, self.BIT_LEN)
        bit=morgan_fps.ToBitString()
        return bit
    
    def calcFingerprintVec(self,smiles):
        """
        smiles: SMILES
        return: vectorized FP
        """
        bit=self.calcFingerprintBit(smiles)
        
        #just convert bit to Hexadecimal.
        #TODO there may be a better way to express chemical info by vectors
        bitList=[bit[i*self.BIT_DIV:(i+1)*self.BIT_DIV] for i in range(int(self.BIT_LEN/self.BIT_DIV))]    
        vec=[int("0b"+b,0) for b in bitList]
        
        return vec

    #prepare dict of vectors of chemicals
    def prepCompDict(self):
        print("calculate fingerprints")
        compDict={}

        for smiles in tqdm(self.smilesList):
            try:
                vec=self.calcFingerprintVec(smiles)
                compDict[smiles]=vec
            except:
                print("error ",smiles)
                
        self.compDict=compDict
    
    #convert compound dict to dataframe     
    def convCompDf(self):
        df=pd.DataFrame(self.compDict).T
        columnNum=["N"+str(i) for i in range(df.shape[1])]
        columnName=["SMILES"]
        columnName.extend(columnNum)
        df=df.reset_index()
        df.columns=columnName
        self.descDf=df

    #automatically process compound file
    def processCompFile(self,compFilename):
        """
        compFilename: path to compound csv file
        """
        self.LoadSMILESList(compFilename)
        self.prepCompDict()
        self.convCompDf()

        
#get column names of a dataframe
def searchColumnNames(df,name):
    res=[]
    for i in df.columns:
        if  i.find(name)!=-1:
            res.append(i)
    return res

#get list of smiles from compound csv
def getSMILESList(compFilename):
    """
    compFilename: path to compound csv (with columns of SMILES***)
    return: list of smiles
    """
    raw_data=pd.read_csv(compFilename)
    sm_columns=searchColumnNames(raw_data,"SMILES")

    smiles_list=[]
    for i in sm_columns:
        smiles=raw_data[i].values
        for j in smiles:
            smiles_list.append(j)

    smiles_list=list(set(smiles_list))
    print("number of smiles: ",len(smiles_list))
    
    return smiles_list
 