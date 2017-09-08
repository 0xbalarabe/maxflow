__author__ = 'balaogbeha'

import networkx as nx
import numpy as np


class Graph(object):
    """
    This class encapsulates a Graph object.
    """

    def __init__(self, nx_graph):
        """
        Initialises a Graph object.

        :return: Graph object
        """

        self.graph = nx_graph

        self.residual_graph = nx_graph

        self.node_order = self.graph.nodes()

        self.edge_order = self.graph.edges()

        self.num_nodes = len(self.graph.nodes())

        self.num_edges = len(self.graph.edges())

    def compute_residual_graph(self, flow):
        """
        Updates a given residual graph with a flow.

        :param graph: A graph object
        :type graph: src.graph

        :param flow: A flow.
        :type flow: numpy.array

        :return: residual_graph: An updated residual graph.
        :rtype src.graph
        """

        for (a, b, c) in self.residual_graph.edges(data=True):

            edge_index = self.edge_order.index((a, b))

            # Update forward residual capacity.
            c['capacity'][0] -= flow[edge_index]

            # Update backward residual capacity.
            c['capacity'][1] += flow[edge_index]

            # Update residual capacity.
            c['residual_capacity'] = min(c['capacity'])

    def compute_congestion_vector(self, flow):
        """
        Computes the congestion vector given a flow.

        :param graph: A graph object
        :type graph: src.graph

        :param flow: A flow.
        :type flow: numpy.array

        :return: congestion_vector: The congestion vector.
        :rtype numpy.array
        """

        congestion_vector = np.zeros(self.num_edges)

        for (u, v, attr) in self.residual_graph.edges(data=True):

            edge_index = self.edge_order.index((u, v))

            congestion_vector[edge_index] = flow[edge_index]/attr['residual_capacity']

        return congestion_vector

    def compute_edge_conductance(self):
        """
        Computes the conductance for every edge in the graph.

        :param graph: A graph object
        :type graph: src.graph

        :return: graph: The updated graph
        :rtype src.graph
        """

        for (u, v, attr) in self.residual_graph.edges(data=True):

            forward = ((attr['capacity'][0] ** 2) ** -1)

            backward = ((attr['capacity'][1] ** 2) ** -1)

            attr['weight'] = (forward + backward) ** -1

    def laplacian(self):
        """
        Computes the laplacaian of the graph.

        :param graph: A graph object
        :type graph: src.graph

        :return: graph: The updated graph
        :rtype numpy.array
        """
        return nx.laplacian_matrix(self.residual_graph, weight='weight', nodelist=self.node_order).todense()

    def s_t_demands(self):
        """
        Computes the s-t demands for a graph.

        :param graph: A graph object
        :type graph: src.graph

        :return: graph: The updated graph
        :rtype numpy.array
        """
        demands = np.zeros(self.num_nodes)

        demands[self.node_order.index('s')] = -1

        demands[self.node_order.index('t')] = 1

        return demands

    def compute_flow(self, potentials):
        """
        Computes a flow from the given set of vertex potentials.

        :param graph: A set of vertex potentials
        :type graph: numpy.array

        :return: flow: The corresponding flow
        :rtype numpy.array
        """

        flow = np.zeros(self.num_edges)

        for (u, v, attr) in self.residual_graph.edges(data=True):

            index_u = self.node_order.index(u)

            index_v = self.node_order.index(v)

            edge_index = self.edge_order.index((u, v))

            potential_u = potentials[index_u]

            potential_v = potentials[index_v]

            flow[edge_index] = (potential_v - potential_u) * attr['weight']

        return flow

    def compute_correction(self, flow, embedding):
        """
        Computes the first-order correction to the primal dual coupling of a
        primal-dual solution.

        :param flow: A flow
        :type flow: numpy.array

        :return: correction: The required correction
        :rtype numpy.array
        """

        correction = np.zeros(self.num_edges)

        delta = self.delta(embedding)

        phi = self.phi(flow)

        for (u, v, attr) in self.residual_graph.edges(data=True):

            edge_index = self.edge_order.index((u, v))

            forward = ((attr['capacity'][0] ** 2) ** -1)

            backward = ((attr['capacity'][1] ** 2) ** -1)

            correction[edge_index] = (delta[edge_index] - phi[edge_index])/(forward + backward)

        return correction

    def delta(self, embedding):

        delta = np.zeros(self.num_edges)

        for (u, v, attr) in self.residual_graph.edges(data=True):

            index_u = self.node_order.index(u)

            index_v = self.node_order.index(v)

            edge_index = self.edge_order.index((u, v))

            delta[edge_index] = embedding[index_v] - embedding[index_u]

        return delta

    def phi(self, flow):

        phi = np.zeros(self.num_edges)

        for (u, v, attr) in self.residual_graph.edges(data=True):

            edge_index = self.edge_order.index((u, v))

            phi[edge_index] = (1.0/(attr['capacity'][0])) - (1.0/(attr['capacity'][1]))

        return phi

    def arc_boosting(self):
        """
        Performs the arc boosting opeation on our graph.

        :param flow: A flow
        :type flow: numpy.array

        :return: correction: The required correction
        :rtype numpy.array
        """

        return None
