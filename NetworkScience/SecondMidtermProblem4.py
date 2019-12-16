# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""Description:
a) Create a network from 100 cliques, where the clique size varies between 5 and 10 randomly!
b) Create a ring from these cliques! The longest one from all shortest paths of the ring should be 200 steps long.
c) Select 10 nodes randomly, and simulate an SI model! The 10 selected nodes are the initially infected nodes and other nodes are healthy at the beginning
. Draw the fraction of infected nodes step-by-step, when the infection probability is 0.02!
"""


#Imports
import networkx as nx
import collections
import matplotlib.pyplot as plt
from itertools import chain






