__author__ = 'balaogbeha'

import networkx as nx


def read_graph(file):

    g = nx.Graph()
    v = 0

    with open(file) as _file:

        for line in _file.readlines():

            line = line.split()

            if line[0] == 'v':
                v = int(line[1])

            if line[0] == 'e':

                print(line[1], line[2], line[3], line[4])

                g.add_edge(line[1], line[2], weight=0, capacity=[int(line[3]), int(line[4])])

    return (g, v)
