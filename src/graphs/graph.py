__author__ = 'balaogbeha'

import scipy as sp
import networkx as nx


def compute_adjacency_matrix(graph):
    return 0


def compute_degree_matrix(graph):
    return 0


def compute_laplacian_matrix(graph):

    d = compute_degree_matrix(graph)

    a = compute_adjacency_matrix(graph)

    return d - a



def compute_residual_graph(graph, flow):
    pass

