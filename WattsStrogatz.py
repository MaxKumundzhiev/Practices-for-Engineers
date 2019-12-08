# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""SmallWorldNetwork contains the parts of a solution to the
small-world network problem in Sethna's book that relate specifically
to the Watts-Strogatz small world networks. The more general graph
algorithms for path lengths and betweenness are in Networks.py."""

# ***** Start by reading the exercise "SixDegreesOfSeparation.pdf"   ***** #
# ***** from SmallWorld.html in                                      ***** #
# ***** www.physics.cornell.edu/~myers/teaching/ComputationalMethods/ComputerExercises/   ***** #

# ***** Then define the general-purpose UndirectedGraph class        ***** #
# ***** using NetworksHints.py (renamed Networks.py), or import your ***** #
# ***** answers previously written for Percolation.                  ***** #

# ***** Then return here to build some small-world networks          ***** #

import random
import os
import scipy
import pylab
#import NetGraphics
#import MultiPlot

# Import your network definitions
import networkx as Networks
#import imp
#imp.reload(Networks)  # Helps with ipython %run command

# Small world and ring graphs


def MakeRingGraph(num_nodes, Z):
    """
    Makes a ring graph with Z neighboring edges per node.
    """
    g = Networks.Graph()
    if Z / 2. != Z / 2:
        raise ValueError("must specify even number of edges per node")
    for i in range(num_nodes):
        for di in range(1, int(Z / 2) + 1):
            j = (i + di) % num_nodes
            g.add_edge(i, j)
    print(g)
    return g


def AddRandomEdges(graph, num_edges_tried):
    """Attempts to add num_edges_tried random bonds to a graph. It may add
    fewer, if the bonds already exist."""
    nodes = graph.GetNodes()
    for n in range(num_edges_tried):
        node1 = random.choice(nodes)
        node2 = random.choice(nodes)
        graph.AddEdge(node1, node2)


def MakeSmallWorldNetwork(L, Z, p):
    """
    Makes a small--world network of size L and Z neighbors,
    with p*Z*L/2 shortcuts added.  This is the Watts-Newman variant
    of the original Watts-Strogatz model.  The original model
    used a rewiring technique, replacing a randomly selected short-range
    bond with a randomly-selected long-range shortcut.  The Watts-Newman
    model keeps all short-range bonds intact, and adds p*Z*L/2 random
    shortcuts.  This revised model is both simpler to treat analytically
    (see the renormalization group analysis by Watts and Newman) and
    avoids the potential for subgraphs to become disconnected from
    one another due to rewiring.
    """
    g = MakeRingGraph(L, Z)
    # should this be a Poisson process with mean L*z*p/2?
    num_shortcuts = int(round(L * Z * p / 2.))
    AddRandomEdges(g, num_shortcuts)
    return g


def SmallWorldSimple(L, Z, p):
    """
    Generate and display small world network. Creates a graph g using
    MakeSmallWorldNetwork, and uses the NetGraphics command
    DisplayCircleGraph, with only the mandatory argument g. Returns g.
    """
    g = MakeSmallWorldNetwork(L, Z, p)
    im = NetGraphics.DisplayCircleGraph(g)
    return g

# ***** After creating, displaying, and debugging your small world   ***** #
# ***** graphs, go to Networks.py and develop and debug the routines ***** #
# ***** for finding path lengths in graphs. (They are put in         ***** #
# ***** Networks.py because they in principle could be used for any  ***** #
# ***** graph. Then return here to study the scaling properties of   ***** #
# ***** path lengths in small-world networks.                        ***** #


def MakePathLengthHistograms(L=100, Z=4, p=0.1):
    """
    Plots path length histograms for small world networks.
    Find list of all lengths
    Use pylab.hist(lengths, bins=range(max(lengths)), normed=True) """
    histograms = []
    g = MakeSmallWorldNetwork(L, Z, p)
    lengths = list(Networks.FindAllPathLengths(g).values())
    pylab.hist(lengths, normed=True, bins=list(range(max(lengths))))
    pylab.show()


def FindAverageAveragePathLength(L, Z, p, numTries):
    """Finds mean and standard deviation for path length between nodes,
    for a small world network of L nodes, Z bonds to neighbors,
    p*Z*L/2 shortcuts, averaging over numTries samples"""
    total = 0.
    total2 = 0.
    for i in range(numTries):
        g = MakeSmallWorldNetwork(L, Z, p)
        ell = Networks.FindAveragePathLength(g)
        total += ell
        total2 += ell * ell
    totalBar = total / numTries
    if (numTries > 1):
        totalVariance = total2 / numTries - totalBar * totalBar
        totalSigmaMean = scipy.sqrt(totalVariance / (numTries - 1))
    else:
        totalSigmaMean = None
    return totalBar, totalSigmaMean


def GetPathLength_vs_p(L, Z, numTries, parray):
    """Calculates array of mean pathlengths and sigmas for small
    world networks; returns pathlengths and sigmas"""
    pathlengths = []
    sigmas = []
    for p in parray:
        pathlength, sigma = FindAverageAveragePathLength(L, Z, p, numTries)
        pathlengths.append(pathlength)
        sigmas.append(sigma)
    pathlengths = scipy.array(pathlengths)
    sigmas = scipy.array(sigmas)
    return pathlengths, sigmas


def PlotPathLength_vs_p(L, Z, numTries=2,parray=10.**scipy.arange(-3., 0.001, 0.25)):
    """Plots path length versus p"""
    pathlengths, sigmas = GetPathLength_vs_p(L, Z, numTries, parray)
    if numTries > 2:
        pylab.errorbar(parray, pathlengths, yerr=sigmas)
    else:
        pylab.plot(parray, pathlengths)
    pylab.semilogx()
    pylab.show()


def PlotScaledPathLength_vs_pZL(LZarray, numtries=2,pZLarray=10.**scipy.arange(-1., 2.001, 0.25)):
    """
    PlotScaledPathLength_vs_pZL(((L1,Z1),(L2,Z2),...), numtries,
                                   [pZLa,pZLb,pZLc...])
    will plot the scaled path length for small world networks of size Li and
    neighbors Zi, at scaled rewiring probabilities pZLa, pZLb, ...
    Uses either MultiPlot.py to do the scaling, or rescales by hand, depending
    on the implementation chosen.
    To rescale, p is multiplied by Z*L and the mean path length ell is
    multiplied by 2*Z/L.
    """

    pathlengthBar = {}
    pathlengthSigma = {}
    pdata = {}
    for L, Z in LZarray:
        # Shift evaluated points to good, scaled range
        pdata[L, Z] = pZLarray / (Z * L)
        pathlengthBar[L, Z], pathlengthSigma[L, Z] = \
            GetPathLength_vs_p(L, Z, numtries, pdata[L, Z])
    pylab.figure(1)
    MultiPlot.MultiPlot(pdata, pathlengthBar,
                        xform='p->p', yform='ell->ell',
                        yerrdata=pathlengthSigma, showIt=False)
    pylab.semilogx()
    pylab.figure(2)
    MultiPlot.MultiPlot(pdata, pathlengthBar,
                        xform='p->p*Z*L', yform='ell->(2*ell*Z)/L',
                        yerrdata=pathlengthSigma,
                        yerrform='sig->(2*sig*Z)/L',
                        keyNames=('L', 'Z'),
                        loc=3, showIt=False)
    pylab.semilogx()
    pylab.show()

# ***** Clustering coefficient was calculated in the original small  ***** #
# ***** world paper, but is not assigned (optional) in this exercise.***** #


def FindAverageClusteringCoefficient(L, Z, p, numTries):
    """Finds clustering coefficient for small world graph"""
    total = 0.
    total2 = 0.
    for i in range(numTries):
        g = MakeSmallWorldNetwork(L, Z, p)
        c = Networks.ComputeClusteringCoefficient(g)
        total += c
        total2 += c * c
    totalBar = total / numTries
    if (numTries > 1):
        totalVariance = total2 / numTries - totalBar * totalBar
        totalSigmaMean = scipy.sqrt(totalVariance / (numTries - 1))
    else:
        totalSigmaMean = None
    return totalBar, totalSigmaMean


def GetClustering_vs_p(L, Z, numTries, parray):
    clustering = []
    sigmas = []
    for p in parray:
        cluster_coeff, sigma = FindAverageClusteringCoefficient(L, Z, p,
                                                                numTries)
        clustering.append(cluster_coeff)
        sigmas.append(sigma)
    clustering = scipy.array(clustering)
    sigmas = scipy.array(sigmas)
    return clustering, sigmas


def PlotClustering_vs_p(L, Z, numTries,parray=10.**scipy.arange(-3., 0.001, 0.1)):
    clustering, sigmas = GetClustering_vs_p(L, Z, numTries, parray)
    pylab.errorbar(parray, clustering, yerr=sigmas)
    pylab.semilogx()
    pylab.show()


def PlotWattsStrogatzFig2(L, Z, numTries,parray=10.**scipy.arange(-4, 0.001, 0.25)):
    """Duplicate Watts and Strogatz Figure 2: rescale vertical axes"""
    clustering, csigmas = GetClustering_vs_p(L, Z, numTries, parray)
    g = MakeSmallWorldNetwork(L, Z, 0)
    c0 = (Networks.ComputeClusteringCoefficient(g))
    ell0 = Networks.FindAveragePathLength(g)
    pathlengths, psigmas = GetPathLength_vs_p(L, Z, numTries, parray)
    if numTries > 0:
        pylab.errorbar(parray, clustering / c0, yerr=csigmas / c0)
        pylab.errorbar(parray, pathlengths / ell0, yerr=psigmas / ell0)
    else:
        pylab.plot(parray, clustering / c0)
        pylab.plot(parray, pathlengths / ell0)
    pylab.semilogx()
    pylab.show()

# ***** Again, go to Networks.py to generate and debug your          ***** #
# ***** algorithms for measuring Betweenness. (The algorithms are    ***** #
# ***** described not in the exercise writeup, but in the original   ***** #
# ***** papers by Mark Newman and Michelle Girvan.                   ***** #
# ***** Use them on small-world networks here.   ***** #


def TestBetweennessSimple():
    """
    Makes a simple graph for which one can calculate the betweenness by
    hand, to check your algorithm.
    """
    g = Networks.UndirectedGraph()
    g.AddEdge(0, 1)
    g.AddEdge(0, 2)
    g.AddEdge(1, 3)
    g.AddEdge(2, 3)
    g.AddEdge(3, 4)
    edgeBt, nodeBt = Networks.EdgeAndNodeBetweenness(g)
    return g, edgeBt, nodeBt


def SmallWorldBetweenness(L, Z, p, dotscale=4, linescale=2, windowMargin=0.02):
    """
    Display small world network with edge and node betweenness,
    using NetGraphics routine DisplayCircleGraph, passing in arguments
    for edge-weights and node_weights. Passes through the arguments for
    dotscale, linescale, and windowMargin, to fine-tune the graph
    """
    g = MakeSmallWorldNetwork(L, Z, p)
    edgeBt, nodeBt = Networks.EdgeAndNodeBetweenness(g)
    im = NetGraphics.DisplayCircleGraph(g, edgeBt, nodeBt,
                                        dotscale=dotscale,
                                        linescale=linescale,
                                        windowMargin=windowMargin)
    return g


def yesno():
    response = input('    Continue? (y/n) ')
    if len(response) == 0:        # [CR] returns true
        return True
    elif response[0] == 'n' or response[0] == 'N':
        return False
    else:                       # Default
        return True


def demo():
    """Demonstrates solution for exercise: example of usage"""
    print("Small World Demo")
    print("  Small World Network 20 sites, Z=4, p = 0.2")
    SmallWorldSimple(20, 4, 0.2)
    if not yesno():
        return
    print("  Small World Network 1000 sites, Z=2, p = 0.05")
    SmallWorldSimple(1000, 2, 0.05)
    if not yesno():
        return
    print("  Scaled Path Length vs. pZL")
    PlotScaledPathLength_vs_pZL(((100, 2), (100, 4), (200, 2), (200, 4)))
    if not yesno():
        return
    print("  Watts-Strogatz Figure 2")
    PlotWattsStrogatzFig2(40, 4, 4)
    if not yesno():
        return
    print("  Betweenness, 20, 4, 0.1")
    SmallWorldBetweenness(50, 2, 0.1)

#if __name__ == "__main__":
    #demo()

MakeRingGraph(100, 3)