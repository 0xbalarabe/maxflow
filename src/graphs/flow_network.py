__author__ = 'balaogbeha'

import networkx as nx


class FLowNetwork(object):

    def __init__(self):
        self.graph = nx.Graph()
        self.residual_graph = self.graph
        self.source = 0
        self.sink = 0

    def set_source(self, source):
        self.source = source

    def set_sink(self, sink):
        self.sink = sink

    def compute_laplacian(self):
        return nx.laplacian_matrix(self.graph).todense()

    def augment_flow(self, flow):
        # for (u,v,w) in
        return None

    def initialise_residual_graph(self):
        self.residual_graph = self.graph
