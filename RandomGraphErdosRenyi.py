# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


import networkx as nx
import matplotlib.pyplot as plt
import random

def display_graph(G, i, ne):
    pos = nx.spring_layout(G)
    if i == '' and ne == '':
        new_node = []
        rest_nodes = G.nodes()
        new_edges = []
        rest_edges = G.edges()
    elif i == '':
        # new_node = [i]
        # rest_nodes = list(set(G.nodes()) - set(new_node))
        rest_nodes = G.nodes ()
        new_edges = ne
        rest_edges = list(set(G.edges()) - set(new_edges) - set([(b,a) for (a,b) in new_edges]))
    #nx.draw_networkx_nodes(G, pos, nodelist=new_node, node_color='g')
    nx.draw_networkx_nodes(G, pos, nodelist=rest_nodes, node_color='r')
    nx.draw_networkx_edges(G, pos, nodelist=new_edges, edge_color='g', style='dashdot')
    nx.draw_networkx_edges(G, pos, nodelist=rest_edges, node_color='r')
    plt.show()



def plot_degree_distribution(G):
    all_degrees = [val for (node, val) in G.degree()]
    unique_degrees = list(set(all_degrees))
    unique_degrees.sort()
    count_of_degrees = []

    for i in unique_degrees:
        c = all_degrees.count(i)
        count_of_degrees.append(c)

    print(unique_degrees)
    print(count_of_degrees)


    plt.plot(unique_degrees, count_of_degrees, '-ro')
    plt.xlabel('Degrees')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    plt.show()



def erdos_renyi(G, p):
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                r = random.random()
                if r <= p:
                    G.add_edge(i, j)
                    ne = [(i, j)]
                    display_graph(G, '', ne)
                else:
                    ne = []
                    display_graph (G, '', ne)
                    continue


def main():
    n = int(input('Enter number of nodes: '))
    p = float (input('Enter probability: '))
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    display_graph(G, '', '')
    erdos_renyi(G, p)
    plot_degree_distribution(G)


main()


