"""
configs for preparing data
"""

import chainer.links as L

class Config:
    def __init__(self):
        
        """
this is just a memo for author to launch bert...
conda activate chem
cd /media/kh/python/bert
bert-serving-start -model_dir uncased_L-24_H-1024_A-16
        """
        
        #if true, word vectors will be calculated by BERT (and save them as dict). if false, load dict info.
        # this must become true whenever new words appear in the databases
        self.initWordEncoder=False
        
        #maximum dimension of vectors will be CATEGORY_DIM+VALUE_DIM
        self.CATEGORY_DIM=4
        self.VALUE_DIM=60
        
        
        #category IDs
        self.ID_OTHERS=0
        self.ID_COMPOUNDS=1
        self.ID_VALUES=2
        
        #it can be 3. but currently 4 (due to the historical reason. this may not affect results significantly)
        self.NUM_CATEGORY=4

        self.FINGERPRINT_DIM=2048
        
        self.genre=["all"]
        
        #GPU mode for ML
        self.GPUMode=True


        if self.GPUMode:
            self.gpu_device = 0
        else:
            self.gpu_device=-1
    

        #default parameters to be set as problems
        self.targetParams=["ionic conductivity","electric conductivity",
                           "absolute molar magnetic susceptibility",
                            "absolute standard enthalpy of formation",
                            "amorphous density",
                            "amorphous thermal conductivity",
                            "band gap",
                            "simulated band gap",
                            "boiling temperature",
                            "simulated highest occupied molecular orbital",
                            "simulated lowest unoccupied molecular orbital",
                            "broad peak",
                            "converted oxidation potential (v vs nhe)",
                            "converted oxidation/reduction potential (v vs nhe)",
                            "converted reduction potential (v vs nhe)",
                            "crystalline density",
                            "crystalline thermal conductivity",
                            "density",
                            "dipole moment",
                            "simulated dipole moment",
                            "electric conductivity",
                            "flash temperature",
                            "glass expansivity",
                            "glass transition temperature",
                            "heat capacity",
                            "ionic conductivity",
                            "ionization energy",
                            "liquid expansivity",
                            "liquid heat capacity",
                            "melting enthalpy",
                            "melting temperature",
                           # 
                           #"molar concentration",
                            "molar heat capacity",
                            "molar solubility",
                            "molar volume",
                           # 
                           #"number average molecular weight",
                            "oxidation potential (v vs ag/ag+)",
                            "oxidation potential (v vs fc/fc+)",
                            "oxidation potential (v vs li/li+)",
                            "oxidation potential (v vs nhe)",
                            "oxidation potential (v vs sce)",
                            "oxidation potential (v vs she)",
                            "oxidation/reduction potential (v vs li/li+)",
                            "oxidation/reduction potential (v vs nhe)",
                            "oxidation/reduction potential (v vs she)",
                            "partition coefficient",
                            "permittivity",
                            "pka",
                            "polarizability",
                           #"polydispersity",
                            "reduction potential (v vs ag/ag+)",
                            "reduction potential (v vs fc/fc+)",
                            "reduction potential (v vs li/li+)",
                            "reduction potential (v vs nhe)",
                            "reduction potential (v vs sce)",
                            "reduction potential (v vs she)",
                            "refractive index",
                            "simulated biodegrability",
                            "simulated boiling temperature",
                            "simulated melting temperature",
                            "simulated partition coefficient",
                            "solid heat capacity",
                            "solubility parameter",
                            "surface tension",
                            "thermal conductivity",
                            "uv cutoff",
                            "vapor pressure",
                            "viscosity",
                            "decomposition temperature",
                           #"weight percent"
                          
                          ]
        
        self.targetParamMode=True
        
        #embedding of category info.
        self.categoryEmbed = L.EmbedID(self.NUM_CATEGORY,self.CATEGORY_DIM)
