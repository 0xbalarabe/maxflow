import scipy as sp
import numpy as np
import networkx as nx

__author__ = 'balaogbeha'

g = nx.MultiGraph()

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
g.add_edge(1, 5, weight=0, capacity=[2, 2])
g.add_edge(1, 't', weight=0, capacity=[2, 1])
g.add_edge(1, 2, weight=0, capacity=[1, 1])
g.add_edge(2, 4, weight=0, capacity=[2, 1])
g.add_edge(3, 4, weight=0, capacity=[2, 2])
g.add_edge(4, 't', weight=0, capacity=[4, 2])
g.add_edge(5, 't', weight=0, capacity=[2, 2])


# h = nx.Graph()
#
# h.add_node('s')
# h.add_node(1)
# h.add_node(2)
# h.add_node(3)
# h.add_node(4)
# h.add_node(5)
# h.add_node('t')
#
# h.add_edge('s', 1, weight=0, capacity=5)
# h.add_edge('s', 2, weight=0, capacity=1)
# h.add_edge('s', 3, weight=0, capacity=3)
# h.add_edge(1, 5, weight=0, capacity=2)
# h.add_edge(1, 't', weight=0, capacity=2)
# h.add_edge(1, 2, weight=0, capacity=1)
# h.add_edge(2, 4, weight=0, capacity=2)
# h.add_edge(3, 4, weight=0, capacity=2)
# h.add_edge(4, 't', weight=0, capacity=4)
# h.add_edge(5, 't', weight=0, capacity=2)
#
# print(nx.maximum_flow(h,g.nodes()[0], g.nodes()[6], capacity='capacity'))

nodelist = ['s', 1, 2, 3, 4, 5, 't']

edgelist = [('s', 1), ('s', 2), ('s', 3), (1, 5), (1, 't'), (1, 2), (2, 4), (3, 4), (4, 't'), (5, 't')]

for (u, v, attr) in g.edges(data=True):
    attr['weight'] = 1/(((attr['capacity'][0] ** 2) ** -1) + ((attr['capacity'][1] ** 2) ** -1))

print(nx.laplacian_matrix(g).todense())


primal = 0
dual = 0



# print(g.edges(data=True))
#
# for (a, b, c) in g.edges(data=True):
#     print(a,b,c)
#     print(c['weight'])

laplacian = nx.laplacian_matrix(g, nodelist=nodelist).todense()
# demand = np.array([0, 0, 0, 0, 0, 0, 0])[np.newaxis].T
demand = np.array([-8, 0, 0, 0, 0, 0, 8])[np.newaxis].T
#
# incidence = nx.incidence_matrix(g, nodelist=nodelist, edgelist=edgelist, oriented=True).todense()
#
# weights = nx.attr_matrix(g, edge_attr='weight', rc_order=nodelist)
#
# print('weights dim: ' + str(weights.shape))
#
potentials = sp.linalg.lstsq(laplacian, demand)[0]

# print(potentials[0])

max_flow = np.zeros(10)

flow = np.zeros(10) #np.array([.0, .0, .0, .0, .0, .0, .0, .0, .0, .0])

for (a, b, c) in g.edges(data=True):

    index_a = nodelist.index(a)

    index_b = nodelist.index(b)

    edge_index = edgelist.index((a, b))

    # print(a,b)
    # print('index a', index_a)
    # print('index b', index_b)
    # print('potentials:', potentials[index_b][0] - potentials[index_a][0])
    # print('weight:', c['weight'])
    # print('product:', (potentials[index_b][0] - potentials[index_a][0])* c['weight'] )
    # print('edge index:', edge_index)
    # print()

    flow[edge_index] = (potentials[index_b][0] - potentials[index_a][0]) * c['weight']

print(flow)


residual_cap = np.zeros(10)

for (a, b, c) in g.edges(data=True):

    edge_index = edgelist.index((a, b))

    residual_cap[edge_index] = min(c['capacity'])

print(residual_cap)

congestion = np.zeros(10)

for (a, b, c) in g.edges(data=True):

    edge_index = edgelist.index((a, b))

    congestion[edge_index] = flow[edge_index]/residual_cap[edge_index]

print(congestion)

step_size = 1/(4 * np.linalg.norm(congestion, ord=4))

print(step_size)

max_flow += step_size * flow

print(max_flow)
# wb = weights * incidence
# print(wb * potentials[0])

# incidence = nx.incidence_matrix(g, oriented=True).todense()
# print(incidence)
#
# print('before')
# print(nx.laplacian_matrix(g).todense())

# s = np.array([(1, 2), (2, 3)])
# print(s[1])
#
# for (u,v,attr) in g.edges(data=True):
#
#         print(attr['capacity'])
#
#         attr['weight'] = 1/(((attr['capacity'][0] ** 2) ** -1) + ((attr['capacity'][1] ** 2) ** -1))
#
# print(nx.attr_matrix(g, edge_attr='weight',rc_order=['s',1,2,'t']))
#
# print('after')
#
# ordering = g.nodes()
# print(ordering)
#
# demands = np.zeros(len(g.nodes()))
#
# print(demands)
# demands[ordering.index('s')] = -1
# demands[ordering.index('t')] = 1
#
# print(demands)
# print(nx.laplacian_matrix(g).todense())
#
# for (u,v,d) in g.edges(data=True):
#
#         print(d['capacity'])
#
#         d['weight'] = ((d['capacity'][0] ** 2) ** -1) + ((d['capacity'][1] ** 2) ** -1)
# print(g.edges(data=True)[0][2]['weight'][0])


# A = nx.adjacency_matrix(g).todense()

# L = nx.laplacian_matrix(g).todense()
# p = nx.laplacian_matrix(g)

# print(p)
# print(L)
# print(p*2)

# x = np.array([-7,0,0,0,0,0,7])[np.newaxis].T
# y = np.array([-7,0,0,0,0,0,7])[np.newaxis].T
# print((x+y))

# bb = np.array([6,-4,27])[np.newaxis].T

# pp = np.matrix([[1,1,1],[0,2,5],[2,5,-1]])

# print(x)
# print(sp.linalg.lstsq(L, x))
# print(np.matmul(sp.linalg.pinv(L), x))
