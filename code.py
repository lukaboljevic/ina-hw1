from collections import defaultdict
import numpy as np
import time

def read_graph(graph_label, directed):
    """
    Read and return the graph and it's transpose, if directed = True
    Otherwise, the transpose graph is empty
    """
    graph = {}
    transpose_graph = {}

    with open(f"graphs/{graph_label}", "r") as f:
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
            
            s = line.split()
            node = s[1]
            graph[int(node)] = []
            if directed:
                transpose_graph[int(node)] = []

        # Adding edges
        for line in f:
            left, right = line.split()
            left, right = int(left), int(right)
            
            if right not in graph[left]:
                graph[left].append(right)

            if directed:
                if left not in transpose_graph[right]:
                    transpose_graph[right].append(left)
            else:
                if left not in graph[right]:
                    graph[right].append(left)

    return graph, transpose_graph

#######################################################
##### Problem 1.3 - strongly connected components #####
#######################################################

def dfs(graph, starting_node, visited):
    """
    Perform iterative DFS on the graph, given a starting node.
    Return which nodes have been visited during this search.
    """
    stack = [starting_node]
    visited[starting_node] = True
    reached = []

    while stack:
        node = stack.pop()
        reached.append(node)

        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    
    return reached

def dfs_transpose(graph, starting_node, visited, reached_original, total_computed):
    """
    Perform iterative DFS on the (transpose) graph, given a starting node.
    reached_original is a list of nodes visited/reached during DFS on the 
    original graph, while total_computed is a list of nodes for which a 
    connected component has been computed.

    The fact that a graph and it's tranpose share strongly connected
    components can be used here - intersection of nodes visited during
    DFS on the original, and transpose graph, will give us a SCC.
    We can compute the component right away, i.e. during the search itself.
    """
    stack = [starting_node]
    visited[starting_node] = True
    component = []

    while stack:
        node = stack.pop()
        if node in reached_original:
            # since it was visited during DFS on the original graph,
            # this means this node is in the same strongly connected
            # component
            component.append(node) 
            total_computed[node] = True

        for neighbor in graph[node]:
            if not visited[neighbor] and not total_computed[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    
    return component

def enron_components():
    """
    Do task 1.3, i.e. find the strongly connected components of the 
    "Enron" graph.
    """
    """
    Number of components: 78058
    Largest component: 9164, which is 10.5%
    Computation time: 1172.823 s

    # If I add "total_computed" in the DFS traversal of transpose graph,
    Number of components: 78058
    Largest component: 9164, which is 10.5%
    Computation time: 254.654 s
    """

    graph, transpose_graph = read_graph("enron", True)
    n = len(graph)
    print("Graph read")

    num_components = 0
    largest_component = 0
    visited = np.full((1, n+1), False).flatten()
    transpose_visited = np.full((1, n+1), False).flatten()
    total_computed = np.full((1, n+1), False).flatten()
    start_time = time.perf_counter()

    for i, node in enumerate(graph):
        print(f"Remaining: {n - (i+1)}")

        if not total_computed[node]:
            num_components += 1
            reached = dfs(graph, node, visited.copy())
            component = dfs_transpose(transpose_graph, node, transpose_visited.copy(), reached, total_computed)
        else:
            # don't do useless work
            continue

        largest_component = max(largest_component, len(component))

    print(f"Number of components: {num_components}")
    print(f"Largest component: {largest_component}, which is {round((largest_component / n)*100, 3)}%")
    print(f"Computation time: {round(time.perf_counter() - start_time, 3)} s")

##########################################################
##### Problem 1.5 - 90-percentile effective diameter #####
##########################################################

def bfs_for_diameter(graph, start):
    """
    Perform BFS on the graph, return what was the deepest level
    reached, and which node (one of them) was at that level
    """
    queue = [start]
    n = len(graph)
    visited = np.full((1, n+1), False).flatten()
    visited[start] = True
    level = -1

    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.pop(0)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    diameter_end = neighbor
                    queue.append(neighbor)
        level += 1
    
    return level, diameter_end

def diameter(graph_label, directed):
    """
    Find the diameter of the graph (one of aps_...)
    """
    graph, tg = read_graph(graph_label, directed)
    d1, end1 = bfs_for_diameter(graph, 1)
    d2, end2 = bfs_for_diameter(graph, end1)
    return d2, end1, end2

def bfs(graph, start, distances_dict):
    """
    Perform BFS on the graph, save how many times
    we encountered some distance in the process
    """
    queue = [start]
    n = len(graph)
    visited = np.full((1, n+1), False).flatten()
    visited[start] = True
    distance = {} # for distances between start and every other node
    distance[start] = 0

    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                neighbor_distance = distance[node] + 1
                distance[neighbor] = neighbor_distance
                distances_dict[neighbor_distance] += 1
                queue.append(neighbor)

def effective_diameter_d90(graph_label, directed):
    """
    Calculate the 90-percentile effective diameter on 
    one of aps_... graphs
    """

    graph, tg = read_graph(graph_label, directed)
    n = len(graph)
    distances_dict = defaultdict(int) # save how many times we encountered a particular distance

    for node in graph:
        if node % 100 == 0:
            print(f"({graph_label}) At node {node} out of {n}")

        bfs(graph, node, distances_dict)

    sum_distances = sum(distances_dict.values()) # sum up the number of times each distance was seen
    percentage = round(sum_distances * 0.9)
    current_sum = 0
    for distance, number in distances_dict.items():
        current_sum += number
        # as soon as we sum up to more than 90%, return at which distance we're at
        # and that's our effective diameter
        if current_sum >= percentage:
            return distance

def effective_diameters():
    """
    Calculate the effective diameter of each graph
    """
    graphs = ["aps_2010_2011", "aps_2010_2012", "aps_2010_2013"]
    outfile = "effective diameters.txt"
    for graph in graphs:
        start = time.perf_counter()
        d90 = effective_diameter_d90(graph, False)
        end = time.perf_counter()

        with open(outfile, "a") as f:
            f.write(f"Graph: {graph}\n")
            f.write(f"Time taken to calculate d90: {end - start} seconds\n")
            f.write(f"d90: {d90}\n\n")

def diameters():
    """
    Calculate diameter of each graph
    """
    graphs = ["aps_2010_2011", "aps_2010_2012", "aps_2010_2013"]
    outfile = "diameters.txt"
    for graph in graphs:
        d, node_1, node_2 = diameter(graph, False)
        with open(outfile, "a") as f:
            f.write(f"Graph: {graph}\n")
            f.write(f"Diameter: {d}\n")
            f.write(f"Start node: {node_1}\n")
            f.write(f"End node: {node_2}\n\n")

# effective_diameters()
# diameters()
# enron_components()
