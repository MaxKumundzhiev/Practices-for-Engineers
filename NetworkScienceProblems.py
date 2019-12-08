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
import numpy as np
import random

"""_______________________________________________________________________________________________________________________________________
Exercise 1:
Write pyhton code that generates an N*N random symmetric matrix with a zero diagonal and entires Bernulli random variables;
i.e each entry is 1 probability p and 0 with probability 1 - p;
Note, such matrix is an adjency matrix for Erdos-Renyi graph;
_______________________________________________________________________________________________________________________________________"""

#Solution
def random_symmetric_matrix(n, p):
    """N*N random symmetric matrix with a zero diagonal and entires Bernulli random variables
    n - number of dimensions
    p - probability"""
    y = np.triu((0 + (np.random.rand(n, n) >= (1 - p))), 1)
    y_sum = (y + y.T)
    return y_sum
print(random_symmetric_matrix(10, 0.6))

"""_______________________________________________________________________________________________________________________________________
Exercise 2:
Given a simple graph G with adjency matrix A.
Calculate the clustering coefficient of all the verticies(nodes) of G directly from a matrix A;
_______________________________________________________________________________________________________________________________________"""

#Solution
def clustering_coefficient(n, p):
    """Calculate the clustering coefficient of all the verticies(nodes) of G directly from a matrix A"""
    y = np.triu(0 + (np.random.rand(n, n) >= 1 - p), 1)
    A = np.mat(y + y.T)

    num = np.diag(A ** 3) / 2
    A_cols = np.array(np.sum(A, 0))
    A_cols[A_cols <= 1] = 2

    denom = A_cols * (A_cols - 1) / 2
    clust_coef = num/denom

    return clust_coef
print(clustering_coefficient(10, 0.6))

"""_______________________________________________________________________________________________________________________________________
Answer Exercise 2:
The i-th diagonal of A ** 3 is the number of pairs of verticies that are connected by an edge and
are both connected to vertex i. Then calculated the denominator which is the number of all pairs of verticies connected to the given vertex i.
It's done by summing the columns of adjency matrix of A, thrn calculated the number of combinations;""
_______________________________________________________________________________________________________________________________________"""

"""_______________________________________________________________________________________________________________________________________
Exercise 3
Construct an Erdos-Renyi graph with n = 300 and p = 0.1;
Plot, find clusterng coefficients;
_______________________________________________________________________________________________________________________________________"""

#Solution
def erdos_renyi_cluster_coeff(n, p):
    G = nx.erdos_renyi_graph (n, p)
    nx.draw_networkx (G)
    plt.show ()
    cluster_coef = nx.clustering (G)
    average_cluster_coef = nx.average_clustering (G)
    return (cluster_coef, average_cluster_coef)
print(erdos_renyi_cluster_coeff(300, 0.2))

"""_______________________________________________________________________________________________________________________________________
Answer Exercise 3:
As n is larger as clustering coefficient becomes smaller, closer to fix p.
Clustering coefficient of a vertex is the conditional probability that 2 it's neighbours 
are chosen at a random, are connected by an edge, given that both of them are connected to the given vetrex
_______________________________________________________________________________________________________________________________________"""

"""_______________________________________________________________________________________________________________________________________
Exercise 4
From Exercise 3 -->
 - Construct an Erdos-Renyi graph with n = 300 and p = 0.1; ++
 - Plot, find clusterng coefficients; ++
 
Plot the histogram of vertex degree of E-R graph. What is the distribution? 
_______________________________________________________________________________________________________________________________________"""

#Solution
def erdos_renyi_distribution(n, p):
    y = np.triu (0 + (np.random.rand (n, n) >= 1 - p), 1)
    A = np.mat (y + y.T)
    G = nx.Graph(A)

    vertex_degrees = list(dict(nx.degree(G)).values())
    plt.hist(vertex_degrees, bins=np.max(vertex_degrees) - np.min(vertex_degrees) + 1,
             facecolor='blue', alpha=.75, rwidth=.9)
    plt.grid(True)
    plt.show()
erdos_renyi_distribution(300, 0.1)

"""_______________________________________________________________________________________________________________________________________
Answer Exercise 4:
The distribution of the vertex degree is a normal distribution
The shape of the graph looks as a normal distribution 
_______________________________________________________________________________________________________________________________________"""

"""_______________________________________________________________________________________________________________________________________
Exercise 5
Create a graph from a matrix A, plot it.
How many vertices, edges, connected components does it have? 
What is the it's average clustering coefficient? Plot the histogram of it's vertices degrees.

If the graph would be E-R random graph, what would be the most likely values for n and p?       
_______________________________________________________________________________________________________________________________________"""

