__author__ = 'balaogbeha'

import scipy as sp
import networkx as nx

def compute_laplacian_matrix(graph):

    return nx.laplacian_matrix(graph).todense()

def compute_residual_graph(graph, flow):
    pass

