import sys
import src.util.read as RD
import src.util.graph as GP
import src.electrical_flow.max_flow as MF

__author__ = 'balaogbeha'

filename = sys.argv[1]


def run(filename):
    (graph, value) = RD.read_graph(filename)

    graph = GP.Graph(graph)

    flow = MF.electrical_max_flow(graph, value)

    print(flow)

run(filename)