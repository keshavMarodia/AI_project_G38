import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define the search algorithms

def dfs(G, start, end):
    visited = set()
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node not in visited:
            visited.add(node)
            if node == end:
                return path
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None

def dfs_path(G, start, end):
    edges = list(nx.dfs_edges(G, start))
    if not edges:
        return None
    path = [start] + [v for u, v in edges]
    return path if path[-1] == end else None


def bfs(G, start, end):
    visited = set()
    queue = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        if node not in visited:
            visited.add(node)
            if node == end:
                return path
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None

def bfs_path(G, start, end):
    tree = nx.bfs_tree(G, start)
    if end not in tree:
        return None
    path = [end]
    while path[-1] != start:
        parent = next(tree.predecessors(path[-1]))
        path.append(parent)
    return list(reversed(path))

def manhattan_distance(x, y):
    try:
        x_float = list(map(float, x.split(',')))
        y_float = list(map(float, y.split(',')))
        return sum(map(abs, (a - b for a, b in zip(x_float, y_float))))
    except ValueError:
        # Return infinity for invalid inputs
        return float('inf')

def astar(G, start, end):
    return nx.astar_path(G, start, end, heuristic=nx.manhattan_distance)

def pso(G, start, end, n_particles, max_iter, c1, c2):
    # Define fitness function
    def fitness_func(position):
        return nx.shortest_path_length(G, source=start, target=end, weight='weight', path=position)
    # Initialize particles
    n_nodes = len(G.nodes())
    pop = np.zeros((n_particles, n_nodes), dtype=int)
    for i in range(n_particles):
        pop[i,:] = np.random.permutation(n_nodes)
    vel = np.zeros_like(pop)
    best_positions = pop.copy()
    best_fitness = np.array([fitness_func(p) for p in pop])
    global_best_position = best_positions[best_fitness.argmin()].copy()
    global_best_fitness = best_fitness.min()
    # Update particles
    for i in range(max_iter):
        # Update velocity
        r1 = np.random.uniform(size=pop.shape)
        r2 = np.random.uniform(size=pop.shape)
        vel = 0.5 * vel + c1 * r1 * (best_positions - pop) + c2 * r2 * (global_best_position - pop)
        # Update position
        pop += vel.astype(int)
        pop = np.clip(pop, 0, n_nodes-1)
        # Update fitness
        fitness = np.array([fitness_func(p) for p in pop])
        # Update particle best positions
        mask = fitness < best_fitness
        best_positions[mask,:] = pop[mask,:]
        best_fitness[mask] = fitness[mask]
        # Update global best position
        if best_fitness.min() < global_best_fitness:
            global_best_position = best_positions[best_fitness.argmin()].copy()
            global_best_fitness = best_fitness.min()
    # Return global best position
    return global_best_position

def ga(G, start, end, pop_size, n_generations, mutation_prob):
    # Define fitness function
    def fitness_func(position):
        return nx.shortest_path_length(G, source=start, target=end, weight='weight', path=position)
    # Initialize population
    n_nodes = len(G.nodes())
    pop = np.zeros((pop_size, n_nodes), dtype=int)
    for i in range(pop_size):
        pop[i,:] = np.random.permutation(n_nodes)
    # Update population
    for i in range(n_generations):
        # Select parents
        fitness = np.array([fitness_func(p) for p in pop])
        fitness[fitness == 0] = 1e-6  # avoid division by zero
        probs = 1 / fitness
        probs /= probs.sum()
        parents = np.random.choice(range(pop_size), size=(pop_size, 2), p=probs)
        # Perform crossover
        offspring = np.zeros_like(pop)
        for j in range(pop_size):
            parent1, parent2 = parents[j]
            # Choose crossover point
            cp = np.random.randint(1, n_nodes-1)
            # Perform crossover
            offspring[j,:cp] = pop[parent1,:cp]
            for k in pop[parent2,cp:]:
                if k not in offspring[j,:cp]:
                    offspring[j,cp] = k
                    cp += 1
                if cp == n_nodes:
                    break
            # Mutate offspring
            for k in range(n_nodes):
                if np.random.uniform() < mutation_prob:
                    idx = np.random.randint(n_nodes)
                    offspring[j,k], offspring[j,idx] = offspring[j,idx], offspring[j,k]
        # Replace population with offspring
        pop = offspring.copy()
    # Return best individual
    fitness = np.array([fitness_func(p) for p in pop])
    return pop[fitness.argmin()].tolist()

# Define the graph
G = nx.Graph()
G.add_weighted_edges_from([
    ('A', 'B', 4),
    ('A', 'C', 2),
    ('B', 'C', 1),
    ('B', 'D', 5),
    ('C', 'D', 8),
    ('C', 'E', 10),
    ('D', 'E', 2),
    ('D', 'F', 6),
    ('E', 'F', 3)
])

# Define the start and end nodes
start = 'A'
end = 'F'

# Define the parameters for PSO and GA
n_particles = 50
max_iter = 100
c1 = 2.0
c2 = 2.0
pop_size = 50
n_generations = 100
mutation_prob = 0.1

# Define the search algorithms
search_algorithms = [
    ('DFS', dfs),
    ('BFS', bfs),
    ('A*', astar),
    ('PSO', pso),
    ('GA', ga)
]

# Run the search algorithms and measure their performance
times = []
for name, algorithm in search_algorithms:
    start_time = time.time()
    if name == 'DFS':
        path = dfs_path(G, start, end)
    elif name == 'BFS':
        path = bfs_path(G, start, end)
    elif name == 'PSO':
        path = algorithm(G, start, end, n_particles, max_iter, c1, c2)
    elif name == 'A*':
        path = nx.astar_path(G, start, end, heuristic=manhattan_distance, weight='weight')
    elif name == 'GA':
        path = algorithm(G, start, end, pop_size, n_generations, mutation_prob)
    else:
        path = algorithm(G, start, end)
    end_time = time.time()
    length = nx.shortest_path_length(G, source=start, target=end, weight='weight')
    times.append((name, length, end_time - start_time))

# Plot the accuracy of the search algorithms
fig, ax = plt.subplots()
ax.bar([name for name, _, _ in times], [length for _, length, _ in times])
ax.set_ylabel('Length of shortest path')
ax.set_title('Performance of search algorithms')
plt.show()