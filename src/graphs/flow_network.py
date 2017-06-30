__author__ = 'balaogbeha'

import networkx as nx


class FLowNetwork(object):

    def __init__(self):
        self.graph = nx.Graph()
        self.source = 0
        self.sink = 0

    def set_source(self, source):
        self.source = source

    def set_sink(self, sink):
        self.sink = sink
