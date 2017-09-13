__author__ = 'balaogbeha'

import scipy as sp

def laplacian_solver(laplacian, demand):
    """
    Solves a Laplacian system of equations.

    :param laplacian: The Laplacian matrix of a graph
    :type laplacian: numpy.array

    :param demand: A vector of demands.
    :type demand: numpyp.array

    :return: solution: The solution.
    :rtype: numpy.array
    """

    solution = sp.linalg.lstsq(laplacian, demand)[0]

    return solution
