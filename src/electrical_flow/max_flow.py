"""
This algorithm computes a maximum flow given a flow network.

It is based on the algorithm presented in the paper:

    Name:"Computing Electrical Flows With Augmenting Electrical FLows."
    Author: "Aleksander Madry"
    Published: "21 August 2016"
    Link: "https://arxiv.org/pdf/1608.06016.pdf"

"""

__author__ = 'balaogbeha'

import numpy as np
import networkx as nx

import src.util.graph
import src.util.linalg


def electrical_max_flow(graph, value):
    """
    Compute a maximum flow using augmenting electrical flows.

    :param graph: A graph object
    :type graph: src.graph

    :param value: The value of the flow to be computed.
    :type value: int

    :return: flow if it exists in graph.
    :rtype numpy.array
    """

    """ Tracks primal progress. If primal = 1, then we have successfully found
        a flow with the given value, i.e. F <= F*.
    """
    primal = 0

    """ Embodies a primal-dual solution that is augmented at every iteration.
        The size of each iteration is carefully chosen to maintain the
        coupling.
    """
    solution = {

        'flow': np.zeros(graph.num_edges),

        'embedding': np.zeros(graph.num_nodes)
    }

    graph.compute_residual_graph(solution['flow'])

    while value > 0:

        """ ... AUGMENTING STEP ...
            Computes an augmenting electrical flow that routes a fraction of the
            given value through the graph.
            This flow is induced by the potentials that are the solution to a
            Laplacian system and in general, may not be well coupled.
        """

        print('... AUGMENTING STEP ...')

        graph.compute_edge_conductance()

        laplacian = graph.laplacian()

        demands = value * graph.s_t_demands()

        """ Tracks dual progress. If dual > 2m/(1 - primal), then we have
            failed to find a flow with the given value, i.e. F > F*.
        """
        dual = np.inner(demands, solution['embedding'])

        if dual > ((2*graph.num_edges)/1-primal):
            print('FAIL')
            return 'FAIL'

        potentials = src.util.linalg.laplacian_solver(laplacian, demands)

        flow = graph.compute_flow(potentials)

        congestion = graph.compute_congestion_vector(flow)

        step_size = (33 * np.linalg.norm(congestion, ord=4)) ** -1

        augmenting_flow = step_size * flow

        solution['flow'] = np.add(solution['flow'], augmenting_flow)

        solution['embedding'] = np.add(solution['embedding'], step_size * potentials)

        graph.compute_residual_graph(augmenting_flow)

        value *= (1 - step_size)

        """ ... FIXING STEP ...
            Computes a correction flow that fixes the coupling. This correction
            flow makes some extra primal progress than we have made in our
            augmenting step (even though it is well coupled) and so it
            must be scaled back by exactly the same amount of extra primal
            progress it has made.
            In essence, the fixing step has the effect of computing a
            circulation flow and adding it to the augmenting electrical flow we
            have already computed to reinstate our required coupling.
        """

        print('... FIXING STEP ...')

        flow_correction = graph.compute_correction(solution['flow'], solution['embedding'])

        corrected_flow = np.add(solution['flow'], flow_correction)

        flow_correction_demands = -1 * (nx.incidence_matrix(graph.residual_graph, oriented=True).todense() * flow_correction[np.newaxis].T)

        graph.compute_residual_graph(flow_correction)

        graph.compute_edge_conductance()

        laplacian = graph.laplacian()

        potentials = src.util.linalg.laplacian_solver(laplacian, flow_correction_demands)[:, 0]

        fixing_flow = graph.compute_flow(potentials)

        solution['flow'] = np.add(corrected_flow, fixing_flow)

        solution['embedding'] = np.add(solution['embedding'], potentials)

        graph.compute_residual_graph(fixing_flow)

        print('- - - - - - - - - - - - - - - - - - - - ')
        print()

    print(solution['flow'])
    return solution['flow']


g = nx.Graph()

g.add_node('s')
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_node(4)
g.add_node(5)
g.add_node('t')

g.add_edge('s', 1, weight=0, capacity=[5, 4])
g.add_edge('s', 2, weight=0, capacity=[1, 1])
g.add_edge('s', 3, weight=0, capacity=[3, 2])
g.add_edge(1, 2, weight=0, capacity=[1, 1])
g.add_edge(1, 5, weight=0, capacity=[2, 2])
g.add_edge(1, 't', weight=0, capacity=[2, 1])
g.add_edge(2, 4, weight=0, capacity=[2, 1])
g.add_edge(3, 4, weight=0, capacity=[2, 2])
g.add_edge(4, 't', weight=0, capacity=[2, 2])
g.add_edge(5, 't', weight=0, capacity=[2, 2])

_graph = src.util.graph.Graph(g)

electrical_max_flow(_graph, 6)