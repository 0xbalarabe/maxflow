__author__ = 'balaogbeha'

from termcolor import colored
import networkx as nx
import src.graphs.flow_network as fn


def _get_num_vertices(file, line, line_number):
    line = line.split()

    try:
        return int(line[1])
    except ValueError:
        print_error('1', line_number, file)


def _get_num_edges(file, line, line_number):
    line = line.split()

    try:
        return int(line[1])
    except ValueError:
        print_error('2', line_number, file)


def _get_source_id(file, line, line_number):
    line = line.split()

    try:
        return int(line[1])
    except ValueError:
        print_error('1', line_number, file)


def _get_sink_id(file, line, line_number):
    line = line.split()

    try:
        return int(line[1])

    except ValueError:
        print_error('1', line_number, file)


def _get_edge(file, line, line_number):
    line = line.split()

    try:
        int(line[1])
    except ValueError:
        print_error('1', line_number, file)


options = {
    'n': _get_num_vertices,
    'm': _get_num_edges,
    's': _get_source_id,
    't': _get_sink_id,
    'e': _get_edge
}

error_codes = {
    '1': 'Number of vertices must be an integer.',
    '2': 'Number of vertices undeclared.',
    '3': 'Number of edges must be an integer.',
    '4': 'Number of edges undeclared.',
    '5': 'Source vertex undeclared.',
    '6': 'Source vertex must be in the graph.',
    '7': 'Sink vertex undeclared.',
    '8': 'Sink vertex must be in the graph.',
    '9': 'Malformed edge.'
}


def print_error(error_code, line_number, file):
    print(colored('ERR:', 'red', attrs=['reverse', 'blink']), error_codes[error_code])
    print(colored('IN:', 'red', attrs=['reverse', 'blink']), file)
    print(colored('AT LINE:', 'red', attrs=['reverse', 'blink']), line_number)

flags = {
    'n': False,
    'm': False,
    's': False,
    't': False,
    'e': False,
}

warning_codes = {
    '1': 'Undeclared source. Using default s = 0.',
    '2': 'Undeclared sink. Using default t = n-1.'

}


def read_graph(file):

    network = fn.FLowNetwork()

    graph = nx.Graph()

    graph2 = nx.MultiDiGraph()



    # graph.add_edge(1,2)
    # graph.add_edge(0,2)

    graph.add_edge(0,2, weight=0.5)
    graph.add_edge(0,1, weight=0.9)
    print(list(graph.edges()))

    flags = {
        'n': False,
        'm': False,
        's': False,
        't': False,
        'e': False,
    }

    with open(file) as _file:
        # line_number = 0

        for line in _file.readlines():
            # line_number += 1

            print(line[0])
            # options[line[0]](file, line, line_number)
            var = lambda: print(list(graph.edges()))

read_graph('test.txt')
