import math
import numpy as np
import networkx as nx
import src.util.graph
import src.util.linalg


def improved_electrical_max_flow(graph, value):
    """
    Compute a maximum flow using augmenting electrical flows.

    :param graph: A graph object
    :type graph: src.graph

    :param value: The value of the flow to be computed.
    :type value: int

    :return: flow if it exists in graph.
    :rtype numpy.array
    """

    o_value = value

    max_capacity = graph.max_capacity

    eta = (1.0/14) - ((1.0/7)*math.log(max_capacity, graph.num_edges)) - (-3 * math.log(math.log(max_capacity * graph.num_edges, 10), 10))

    # Tracks primal progress. When primal = 1, then we have computed a maximum
    # flow that has the given value.
    primal = 0

    solution = {

        'flow': np.zeros(graph.num_edges),

        'embedding': np.zeros(graph.num_nodes)
    }

    graph.compute_residual_graph(solution['flow'])

    while primal < 1:

        early_termination = (1 - primal) * value <= (graph.num_edges ** (0.5 - eta))

        if early_termination:
            print(solution['flow'])
            return solution['flow']

        congestion = graph.compute_congestion_vector(solution['flow'])

        boosting = (np.linalg.norm(congestion, ord=3)) > ((graph.num_edges ** (0.5 - eta))/(33 * (1 - primal)))

        if boosting:

            graph.arc_boosting(eta, primal, congestion)

        else:

            """ ... AUGMENTING STEP ... """

            # print('... AUGMENTING STEP ...')

            graph.compute_edge_conductance()

            laplacian = graph.laplacian()

            demands = value * graph.s_t_demands()

            potentials = src.util.linalg.laplacian_solver(laplacian, demands)

            flow = graph.compute_flow(potentials)

            congestion = graph.compute_congestion_vector(flow)

            step_size = (33 * (1 - primal) * np.linalg.norm(congestion, ord=3)) ** -1

            augmenting_flow = step_size * flow

            solution['flow'] = np.add(solution['flow'], augmenting_flow)

            solution['embedding'] = np.add(solution['embedding'], step_size * potentials)

            graph.compute_residual_graph(augmenting_flow)

            value *= (1 - step_size)
            primal = (o_value - value)/o_value

            """ ... FIXING STEP ... """

            # print('... FIXING STEP ...')

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

        print(solution['flow'])

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


improved_electrical_max_flow(_graph, 6)
