# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


"""Trees - un-oriented graph, which is on one hand connected on another do not have loops.

    Properties:
    1. If there is n nodes, that means there are n-1 edges.
    2. Between 2 nodes there is just one path.

    Terminology:
    Noded tree - tree where one of nodes defined as node.
    Node - parent node
    Children - children of parent node

    Ways of representing trees:
    1. Using list (array). We keep nodes values and references for the children of it.
    2. Using adjacency list. We keep for each node list of it's children.
    3. Using more complex representation as dict, where we have multiple keys such as parents, children, references etc.

    Popular tasks:
    1. Find tree height
"""


def tree_height(tree):
    height = 1
    for children_index, children in enumerate(tree):
        height = max(height, 1+tree_height(children_index))
    return height





