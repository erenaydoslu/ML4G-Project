import numpy as np
import networkx as nx
import scipy as sp
from scipy.sparse import csr_array
def generate_parametric_product_graph(s00, s01, s10, s11, A_T, A_N, spatial_graph = None):
    #print("Generating product graph...")
    '''
    I_T = np.identity(len(A_T))
    I_N = np.identity(len(A_N))
    S_diamond = (
        s00 * np.kron(I_T, I_N) +
        s01 * np.kron(I_T, A_N) +
        s10 * np.kron(A_T, I_N) +
        s11 * np.kron(A_T, A_N)
    )
    product_graph = nx.from_numpy_array(S_diamond)
    '''
    I_T = csr_array(np.identity(len(A_T)))
    I_N = csr_array(np.identity(len(A_N)))
    A_T = csr_array(A_T)
    A_N = csr_array(A_N)
    S_diamond = (
        s00 * sp.sparse.kron(I_T, I_N) +
        s01 * sp.sparse.kron(I_T, A_N) +
        s10 * sp.sparse.kron(A_T, I_N) +
        s11 * sp.sparse.kron(A_T, A_N)
    )
    return S_diamond
    '''
    
    # Add features to the product graph
    num_nodes = A_N.shape[0]
    num_timesteps = A_T.shape[0]
    
    for t in range(num_timesteps):
        for node in range(num_nodes):
            original_node = node #node_list[node]  # Adjust if necessary
            new_node = t * num_nodes + node
            product_graph.nodes[new_node]['feature'] = spatial_graph.nodes[original_node]['feature']
    
    print("Product graph generated")
    '''
    return product_graph

