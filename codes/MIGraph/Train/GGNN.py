"""
this class is grate graph neural network(GGNN)
original script was obtained from cheiner-chemistry library (MIT lisence).
it was designed to input chemical structures, but this code can input general graphs
some codes were changed slightly
"""

import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer_chemistry.links import GraphLinear
import chainer_chemistry



class GGNN(chainer.Chain):
    """Gated Graph Neural Networks (GGNN)

    See: Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015).\
        Gated graph sequence neural networks. \
        `arXiv:1511.05493 <https://arxiv.org/abs/1511.05493>`_

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying or not

    """
    #original code has multiple type edges, but this code has only one type edge
    NUM_EDGE_TYPE = 4
    NUM_EDGE_TYPE = 1

    def __init__(self,gpu_device, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=1, concat_hidden=False,
                 weight_tying=True):
        super(GGNN, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        self.gpu_device=gpu_device
        with self.init_scope():
            # Update
            #self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
            self.message_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
                  for _ in range(n_message_layer)]
            )
            self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
            # Readout
            self.i_layers = chainer.ChainList(
                *[GraphLinear(2 * hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def update(self, h, adj, step=0):
        # --- Message & Update part ---
        # (minibatch, atom, ch)
        mb, atom, ch = h.shape
        out_ch = ch
        message_layer_index = 0 if self.weight_tying else step
        m = functions.reshape(self.message_layers[message_layer_index](h),
                              (mb, atom, out_ch, self.NUM_EDGE_TYPE))
        # m: (minibatch, atom, ch, edge_type)
        # Transpose
        m = functions.transpose(m, (0, 3, 1, 2))
        # m: (minibatch, edge_type, atom, ch)

        adj = functions.reshape(adj, (mb * self.NUM_EDGE_TYPE, atom, atom))
        # (minibatch * edge_type, atom, out_ch)
        m = functions.reshape(m, (mb * self.NUM_EDGE_TYPE, atom, out_ch))

        m = chainer_chemistry.functions.matmul(adj, m)

        # (minibatch * edge_type, atom, out_ch)
        m = functions.reshape(m, (mb, self.NUM_EDGE_TYPE, atom, out_ch))
        # Take sum
        m = functions.sum(m, axis=1)
        # (minibatch, atom, out_ch)

        # --- Update part ---
        # Contraction
        h = functions.reshape(h, (mb * atom, ch))

        # Contraction
        m = functions.reshape(m, (mb * atom, ch))

        out_h = self.update_layer(functions.concat((h, m), axis=1))
        # Expansion
        out_h = functions.reshape(out_h, (mb, atom, ch))
        return out_h

    def readout(self, h, h0, step=0):
        # --- Readout part ---
        index = step if self.concat_hidden else 0
        
        g = functions.sigmoid(
            self.i_layers[index](functions.concat((h, h0), axis=2))) \
            * self.j_layers[index](h)
        g = functions.sum(g, axis=1)  # sum along atom's axis
        return g

    #@static_graph
    def __call__(self, atom_array, adj):
        
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix with edge-type
                information

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        # reset state
        self.update_layer.reset_state()
        if atom_array.dtype == self.xp.int32:
            raise "use vector expression"
        else:
            h = atom_array
        
        h0 = functions.copy(h, self.gpu_device)

        g_list = []
        for step in range(self.n_layers):
            h = self.update(h, adj, step)           
            if self.concat_hidden:
                g = self.readout(h, h0, step)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = self.readout(h, h0, 0)
            return g
