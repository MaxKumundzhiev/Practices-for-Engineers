# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""Description:
a) Create a random network of 100 nodes and 500 edges! Explain in 2-3 sentences, what are you expecting for the degree distribution of this network!
b) Write an algorithm, which rewires this network by relocating only one end of a single edge in a step.
c) Using the rewiring routine, change the network step by step such, that finally the nodes with largest degree
centrality must have the following degrees: 24, 21, 19, 19, 15, 14. In the first step, when this condition is satisfied, the program must stop.
d) Plot the degree distribution of the final network!"""


#Imports
import networkx as nx
import collections
import matplotlib.pyplot as plt
from itertools import chain

"""a) Create a random network of 100 nodes and 500 edges! Explain in 2-3 sentences, what are you expecting for the degree distribution of this network!"""

def random_graph(num_nodes, num_edges):
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    print('Random graph is created \n')
    print('Number of nodes {} \nNumber of edges {}'.format(len(G.nodes), len(G.edges)))
    return G

def initial_degree_distribution(Graph):
    G = Graph
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    #print("Degree sequence: ", degree_sequence)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    #print(degreeCount, deg, cnt)
    fig, ax = plt.subplots ()
    plt.bar (deg, cnt, width=0.80, color='b')
    plt.title ("Degree Histogram")
    plt.ylabel ("Count")
    plt.xlabel ("Degree")
    ax.set_xticks ([d + 0.4 for d in deg])
    ax.set_xticklabels (deg)
    # draw graph in inset
    plt.axes ([0.4, 0.4, 0.5, 0.5])
    Gcc = G.subgraph (sorted (nx.connected_components (G), key=len, reverse=True)[0])
    pos = nx.spring_layout (G)
    plt.axis ('off')
    nx.draw_networkx_nodes (G, pos, node_size=20)
    nx.draw_networkx_edges (G, pos, alpha=0.4)
    plt.show()

# print(random_graph(100, 500))
# print(initial_degree_distribution(random_graph(100, 500)))

"""b) Write an algorithm, which rewires this network by relocating only one end of a single edge in a step."""
"""In this case we will use:
Boundary:
networkx.algorithms.boundary.edge_boundary(G, nbunch1, nbunch2=None, data=False, keys=False, default=None)

Edge boundaries are edges that have only one end in the set of nodes.
Node boundaries are nodes outside the set of nodes that have an edge to a node in the set.

Routines to find the boundary of a set of nodes.
An edge boundary is a set of edges, each of which has exactly one endpoint in a given set of nodes (or, in the case of directed graphs, the set of edges whose source node is in the set).
A node boundary of a set S of nodes is the set of (out-)neighbors of nodes in S that are outside S."""

def isolated_nodes(Graph):
    G = Graph
    list_of_isolated_nodes = nx.isolates(G)
    print(list_of_isolated_nodes)

    #res = list(chain.from_iterable(nx.edge_boundary(G, nbunch1).values()))
    #res = nx.edge_boundary(G, nbunch1)



isolated_nodes(random_graph(100, 500))

#Important linl: https://stackoverflow.com/questions/49427320/joining-two-networkx-graphs-on-a-single-edge/49429072#49429072