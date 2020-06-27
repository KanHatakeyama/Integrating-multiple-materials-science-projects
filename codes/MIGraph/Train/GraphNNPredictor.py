"""
this class consists of GGNN and dense layers to predict specific parameters
original script was obtained from cheiner-chemistry library (MIT lisence).
some codes were changed

#TODO: we used GGNN, but conventional models, such as CNN, may be used to input "adjacency matrix" + "node vector".
"""
import chainer
from chainer import cuda
from chainer import datasets
import chainer.links as L
import chainer.functions as F

from GGNN import GGNN
from Config import Config

#settings
CF=Config()
CATEGORY_DIM=CF.CATEGORY_DIM
VALUE_DIM=CF.VALUE_DIM

if CF.genre==["V"]:
    OUT_DIM=1
else:
    OUT_DIM=VALUE_DIM

    
#GPU mode
GPUMode=CF.GPUMode

if GPUMode:
    xp = cuda.cupy
    import cupy
    gpu_device = 0
else:
    gpu_device=-1
    xp = np
    
    
class GraphNNPredictor(chainer.Chain):
    def __init__(self):
        
        #dropout ratio
        self.dr = 0.3
        self._device = None
        self.outFingeprint=False

        super(GraphNNPredictor, self).__init__()
        with self.init_scope():
            
            #GGNN part
            out_dim=VALUE_DIM+CATEGORY_DIM
            hidden_dim=out_dim
            n_layers=4
            
            self.graph_conv0 = GGNN(gpu_device,out_dim=out_dim, hidden_dim=hidden_dim, n_layers=n_layers,concat_hidden=False)

            #MLP part
            MLP_hidden_dim=int((VALUE_DIM+CATEGORY_DIM)/2.0)
            self.l1 = L.Linear(out_dim, MLP_hidden_dim)
            self.l2 = L.Linear(MLP_hidden_dim, MLP_hidden_dim)
            self.l3 = L.Linear(OUT_DIM)
     
                
    def __call__(self, atoms, adjs,t=None):
        
        h0 = self.graph_conv0(atoms, adjs)        
        h1 = F.leaky_relu(self.l1(h0))
        h2 = F.dropout(h1, ratio=self.dr)
        h3 = F.leaky_relu(self.l2(h2))
        h4 = self.l3(h3)
        out_h=h4

        #internal output
        if self.outFingeprint==True:
            ret=h3
        #normal output
        elif t is None:     
            ret= out_h 
        #train mode
        else:
            loss=  F.mean_squared_error(t, out_h)
            ret=loss

            # Report the mean absolute and squared errors.
            chainer.report({
                'loss': loss,
            }, self)

        return ret
    
#utility funcs
def myConcat(trBatch,padding=0):    
    return chainer.dataset.concat_examples(trBatch,padding=0,device =  gpu_device)

#node type info is removed for learning
def formatDataset(dset):
    propList,adjList,targetList,nodeTypeList=zip(*dset)
    return datasets.TupleDataset(propList,adjList,targetList)