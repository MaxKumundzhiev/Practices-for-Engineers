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
    else:
        new_node = [i]
        rest_nodes = list(set(G.nodes()) - set(new_node))
        new_edges = ne
        rest_edges = list(set(G.edges()) - set(new_edges) - set([(b,a) for (a,b) in new_edges]))

    nx.draw_networkx_nodes(G, pos, nodelist=new_node, node_color='g')
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

def add_nodes_barabasi(G, n, m0):
    m = m0-1
    for i in range(m0+1, n+1):
        G.add_node(i)
        degrees = nx.degree(G)
        node_probabilities = {}

        for each in G.nodes():
            node_probabilities[each] = (float)(degrees[each])/sum(dict(degrees).values())
        node_probabilities_cum = []
        prev = 0

        for n, p in node_probabilities.items():
            temp = [n, prev + p]
            node_probabilities_cum.append(temp)
            prev += p

        new_edges = []
        num_edges_added = 0
        target_nodes = []

        while (num_edges_added < m):
            prev_cum = 0
            r = random.random()
            k = 0
            while (not(r > prev_cum) and (r <= node_probabilities_cum[k][1])):
                prev_cum = node_probabilities_cum[k][1]
                k += 1
            target_node = node_probabilities_cum[k][0]
            if target_node in target_nodes:
                continue
            else:
                target_nodes.append(target_node)

            G.add_edge(i, target_node)
            num_edges_added += 1
            new_edges.append((i, target_node))

        print(num_edges_added, ' Edges added')
        display_graph(G, i, new_edges)

    return G


def main():
    n = int(input('Enter number of nodes: '))
    m0 = random.randint(2, n/5)
    G = nx.path_graph(m0)
    display_graph(G, '', '')
    G = add_nodes_barabasi(G, n, m0)


main()


