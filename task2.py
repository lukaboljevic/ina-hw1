from collections import defaultdict
import random
import networkx as nx
import matplotlib.pyplot as plt
from math import factorial, exp, pow

def random_node(graph):
    # Graph will be represented using an edge list
    # So something like [(1, 2), (1, 3)]
    # We select a random edge (1/m for each, prob. I select some node is k_i/m)
    # Then selected a random node from there (probability 1/2 for node i)
    # Total is k_i/2m
    m = len(graph)
    index = random.randrange(0, m) # random edge
    node = random.randrange(0, 2) # random node
    return graph[index][node]

def read_facebook():
    """
    Read the facebook graph
    """
    graph = nx.Graph(name="Facebook")

    with open("graphs/facebook", "r") as f:
        # Skip first 4 lines
        f.readline()
        f.readline()
        f.readline()
        f.readline()

        # Adding nodes
        for line in f:
            line = line.strip()
            if line == "#":
                break
            
            tag, node, label = line.split()
            graph.add_node(int(node) - 1, label=label.strip())
            
        # Adding edges
        for line in f:
            node1, node2 = line.split()
            graph.add_edge(int(node1) - 1, int(node2) - 1)
            graph.add_edge(int(node2) - 1, int(node1) - 1)

    return graph

def degree_distribution(sequence, n):
    """
    Return the degree distribution for this degree sequence
    """
    distribution = defaultdict(int)
    for degree in sequence:
        distribution[degree] += 1
    for degree in distribution:
        distribution[degree] /= n
    return distribution

def plot_stuff():
    """
    Plot all the degree distributions
    """
    graph = read_facebook()
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    """
    graph[i] - dictionary containing neighbors of i as keys
    graph.degree[i] - degree of node i
    graph.nodes() - nodes of the graph
    """
    
    # Facebook degree distribution
    fb_sequence = [graph.degree[node] for node in graph.nodes() if graph.degree[node] > 0]
    fb_distribution = degree_distribution(fb_sequence, n)

    # Erdos-Renyi degree distribution
    average_degree = 2*m / n
    erdos = nx.gnm_random_graph(n, m)
    erdos_sequence = [erdos.degree[node] for node in erdos.nodes() if erdos.degree[node] > 0]
    erdos_distribution = degree_distribution(erdos_sequence, n)

    # Erdos-Renyi theoretical degree distribution
    erdos_theoretical = defaultdict(int)
    for degree in set(erdos_sequence):
        numerator = pow(average_degree, degree) * exp(-average_degree)
        denominator = factorial(degree)
        erdos_theoretical[degree] = numerator / denominator

    # Plot
    plt.style.use("ggplot")
    plt.figure(figsize=(14, 8))
    plt.plot(list(fb_distribution.keys()), list(fb_distribution.values()), 
        'o', color="forestgreen", label="Facebook sample")
    plt.plot(
        list(erdos_distribution.keys()), list(erdos_distribution.values()), 
        'o', color='dodgerblue', label="Erdos Renyi G(n, m)")
    plt.plot(
        list(erdos_theoretical.keys()), list(erdos_theoretical.values()), 
        color='firebrick', label="Erdos Renyi theoretical")
    plt.title("Various degree distributions")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.xlabel("Number of nodes from the sample with this degree")
    plt.subplots_adjust(left=0.052, bottom=0.086, right=0.964, top=0.945)
    plt.legend()
    plt.show()

plot_stuff()