import math
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse import csgraph
import networkx as nx


def incremental_sparsify():
    pass


def greedy_elimination(graph):
    result = graph
    pass


def build_chain(graph, probability):
    graph_1 = graph
    chain = {}

    n = graph.get_number_of_edges

    math.log(2)
    pass


def preconditioned_chebyshev(matrix, rhs_vector, iterations, preconditioner, lambda_min, lambda_max):

    x = 0

    r = rhs_vector

    d = (lambda_max + lambda_min)/2

    c = (lambda_max - lambda_min)/2

    for i in range(1, iterations):
        z = preconditioner(r)

        if i == 1:

            x = z

            alpha = 2/d

        else:

            beta = (c * alpha/2) ^ 2

            alpha = 1/(d-beta)

            x = z + beta * x

        x += alpha * x

        r = rhs_vector - matrix * x

    return x


def recursive_preconditioned_chebyshev(chain, level, rhs_vector, iterations):
    pass


def solve_laplacian(laplacian, rhs_vector, error, probability):

    chain = build_chain(laplacian, probability)

    x = recursive_preconditioned_chebyshev(chain, )

    return x