# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""Description:
Create a grid network of size 6x6, where nodes can be in two states: green or yellow and links can have three labels:
gg, yy and gy depending on the node states at the ends of the link.
If both nodes at the ends are green, the link is labeled with gg.
If both ends are yellow, it has yy label. Otherwise the link is labelled with gy.
a) color 16 nodes as green and 20 nodes as yellow randomly.
b) label the links according the rule above
c) rewire the links by choosing randomly two new nodes for each edge
Estimate the probability, that a link is correctly labeled after rewiring."""


#Imports
import networkx as nx
import collections
import matplotlib.pyplot as plt
import random

"""Create a grid network of size 6x6, 
where nodes can be in two states: green or yellow and links can have three labels"""

RED = 0
BLUE = 1
size = 6

g = nx.generators.grid_2d_graph (size, size)
nodes = [(i, j) for i in range (size) for j in range (size)]
pos = {n: n for n in nodes}

print(nodes)
print(pos)

def checkerboard_pattern(x, y):
    d = {}
    next_color_red = True
    for i in range(x):
        for j in range(y):
            if next_color_red:
                d[(i, j)] = RED
                next_color_red = False
            else:
                d[(i, j)] = BLUE
                next_color_red = True
    return d

def draw(graph, positions, d):
    red_nodes = [n for n, c in d.items () if c == RED]
    blue_nodes = [n for n, c in d.items () if c == BLUE]
    nx.draw_networkx_nodes(graph, positions, nodelist=red_nodes, node_color='g')
    nx.draw_networkx_nodes(graph, positions, nodelist=blue_nodes, node_color='y')
    nx.draw_networkx_edges(graph, positions)
    plt.show()

state = checkerboard_pattern(size, size)
draw(g, pos, state)

def update_in_small_step(g, state):
    nodes = list (g.nodes ())
    random.shuffle (nodes)

    if True:
        node = random.choice (list (g.nodes ()))
        print (node)
        red_count = 0;
        blue_count = 0
        for n in g.neighbors (node):
            # num_neighbors ...
            if state[n] == RED:
                red_count += 1
            elif state[n] == BLUE:
                blue_count += 1
        if 0 < red_count + blue_count:
            if 2 / 3 < red_count / (red_count + blue_count):
                state[node] = RED
            elif 2 / 3 < blue_count / (red_count + blue_count):
                state[node] = BLUE
    return state

state2 = update_in_small_step(g, state)
draw(g, pos, state2)

for i in range(5):
    state2 = update_in_small_step(g, state2)
    draw(g, pos, state2)