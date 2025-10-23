#import necessary libraries
import itertools
import numpy as np
import pandas as pd
import random
import time
import os
import tracemalloc


"""common functions"""
# load triangular matrix
def load_triangular_matrix(file_name):
    file_path = os.path.join('DistanceMatrices', file_name)
    with open(file_path, 'r') as file:
        # Skip the first row
        lines = file.readlines()[1:]
        matrix = [list(map(int, line.split())) for line in lines]
    return matrix

# create a symmetric matrix from a triangular matrix
def make_symmetric(matrix):
    n = len(matrix)
    symmetric_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):  # Only iterate up to the diagonal (inclusive)
            symmetric_matrix[i][j] = symmetric_matrix[j][i] = matrix[i][j]

    return symmetric_matrix

# create a dictionary of distances between cities.
def create_distance_lookup(distance_matrix):
    num_cities = len(distance_matrix)
    distance_lookup = {}

    for i in range(num_cities):
        for j in range(i, num_cities): 
            distance_lookup[(i, j)] = distance_lookup[(j, i)] = distance_matrix[i][j]
    sorted_distance_lookup={k:v for k,v in sorted(distance_lookup.items(),key=lambda item:item[1])}
    return sorted_distance_lookup

#calculate the total distance of a route using the distance dictionary
def total_distance(route, distance_lookup):
    total_dist = 0.0
    num_cities = len(route)

    for i in range(num_cities):
        total_dist += distance_lookup[(route[i], route[(i + 1) % num_cities])]

    return total_dist

"""exhausive algorithm to establish baseline"""
# adapted from https://github.com/norvig/pytudes/blob/main/ipynb/TSP.ipynb

def exhaustive_tsp(DistanceMatrix,distance_lookup):
    tours=itertools.permutations(range(DistanceMatrix.shape[0]))
    shortest_tour=list(min(tours, key= lambda x: total_distance(x,distance_lookup)))
    shortest_tour.append(shortest_tour[0])
    return shortest_tour

""" Nearest Neighbor Algorithm """
# adapted from https://medium.com/@suryabhagavanchakkapalli/solving-the-traveling-salesman-problem-in-python-using-the-nearest-neighbor-algorithm-48fcf8db289a

def nearest_neighbor(distances):
    n = distances.shape[0]
    route = [0] # Start at city A
    visited = set([0])
    while len(visited) < n:
        current_city = route[-1]
        nearest_city = min([(i, distances[current_city][i]) for i in range(n) if i not in visited], key=lambda x: x[1])[0]
        route.append(nearest_city)
        visited.add(nearest_city)
    route.append(0) # Return to city A
    return route


"""The following blocks implement MST to solve TSP"""
# adapted from https://github.com/norvig/pytudes/blob/main/ipynb/TSP.ipynb

#return the first element of a collection
def first(collection): 
    return next(iter(collection))

# create a minimum spanning tree from a set of nodes and distances between them
def mst(DistanceMatrix,distance_lookup):
    vertexes=range(DistanceMatrix.shape[0])
    tree  = {first(vertexes): []} # the first city is the root of the tree.
    links = distance_lookup.keys()
    while len(tree) < len(vertexes):
        (A, B) = first((A, B) for (A, B) in links if (A in tree) ^ (B in tree))
        if A not in tree: (A, B) = (B, A)
        tree[A].append(B)
        tree[B] = []
    return tree

#preorder traversal for a tree
def preorder_traversal(tree, root):
    yield root
    for child in tree.get(root, ()):
        yield from preorder_traversal(tree, child)
        
#create MST then traverse in pre-order to produce a route, then append root to create a cycle
def mst_tsp(DistanceMatrix,distance_lookup):
    Tour=list
    route=Tour(preorder_traversal(mst(DistanceMatrix,distance_lookup),0))
    route.append(0)
    return route


"""The next blocks solve TSP using Genetic Algorithm"""
#adapted from https://github.com/SonnyFixit/Travelling_salesman_problem/tree/main

# initialize a population with random routes
def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

# Tournament selection function
def tournament_selection(population, distances, k):
    selected = random.sample(population, k)
    return min(selected, key=lambda x: total_distance(x, distances))

# PMX crossover function
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    child = parent1[a:b+1]
    child_set = set(child)

    for i in range(size):
        if i < a or i > b:
            gene = parent2[i]
            while gene in child_set:
                idx = parent2.index(gene)
                gene = parent2[(idx + 1) % size]
            child.append(gene)
            child_set.add(gene)

    return child

# Inversion mutation function
def inversion_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    if a > b:
        a, b = b, a
    route[a:b+1] = reversed(route[a:b+1])
    return route

# Exchange mutation function
def exchange_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route

# Function to generate a new population and evaluate their routes
def generate_population_and_evaluate(population, distance_lookup, tournament_size):
    new_population = []

    for _ in range(len(population) // 2):
        parent1 = tournament_selection(population, distance_lookup, tournament_size)
        parent2 = tournament_selection(population, distance_lookup, tournament_size)

        if random.random() < crossover_prob:
            child1 = pmx_crossover(parent1, parent2)
            child2 = pmx_crossover(parent2, parent1)
        else:
            child1, child2 = parent1[:], parent2[:]

        if random.random() < inversion_prob:
            child1 = inversion_mutation(child1)
        if random.random() < inversion_prob:
            child2 = inversion_mutation(child2)
        if random.random() < exchange_prob:
            child1 = exchange_mutation(child1)
        if random.random() < exchange_prob:
            child2 = exchange_mutation(child2)

        fitness_child1 = total_distance(child1, distance_lookup)
        fitness_child2 = total_distance(child2, distance_lookup)

        if fitness_child1 < fitness_child2:
            new_population.append(child1)
            new_population.append(parent2)
        else:
            new_population.append(child2)
            new_population.append(parent1)

    return new_population

# Genetic algorithm with improved calculation
def genetic_algorithm_with_elitism(distance_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, elitism_ratio, distance_lookup):
    population = initialize_population(pop_size, len(distance_matrix))
    elitism_count = int(elitism_ratio * pop_size)

    for generation in range(num_generations):
        new_population = generate_population_and_evaluate(population, distance_lookup, tournament_size)
        new_population.sort(key=lambda x: total_distance(x, distance_lookup))
        
        # Preserve the best individuals from the current population
        elite_individuals = new_population[:elitism_count]
        
        # Generate the rest of the population through genetic operations
        non_elite_population = new_population[elitism_count:]
        offspring_population = generate_population_and_evaluate(non_elite_population, distance_lookup, tournament_size)
        
        # Combine elite and offspring populations to form the next generation
        population = elite_individuals + offspring_population

    best_route = min(population, key=lambda x: total_distance(x, distance_lookup))
    
    return best_route

"""2 opt optimization algorithm"""
def two_opt(initial_route, distance_lookup):
    new_route = initial_route[:]
    new_route.pop()  # Remove the last element (duplicate of the start)
    num_cities = len(new_route)
    
    improvement = True
    while improvement:
        improvement = False
        for i in range(num_cities - 1):
            for j in range(i + 2, num_cities):
                # Check if reversing the segment between i and j improves the route
                A=new_route[i]
                B=new_route[i+1]
                C=new_route[j]
                D=new_route[(j+1)%num_cities]
                
                if distance_lookup[A,C]+distance_lookup[B,D]<\
                    distance_lookup[A,B]+distance_lookup[C,D]:
                        new_route[i+1:j+1] = reversed(new_route[i+1:j+1])
                        improvement = True  
    new_route.append(new_route[0])
    return new_route

"""Nearest Neighbor with 2-opt optimization"""
def NN_two_opt(distance_matrix, distance_lookup):
    best_route=two_opt(nearest_neighbor(distance_matrix),distance_lookup)
    return best_route

"""MST with 2-opt optimization"""
def MST_two_opt(distance_matrix, distance_lookup):
    best_route=two_opt(mst_tsp(distance_matrix,distance_lookup),distance_lookup)
    return best_route


"""code to run and time an algorithm while keeping track of peak memory allocation"""
def run_and_store_results(func,func_name,file_name,distance_lookup,*args):
    tracemalloc.start()
    start_time = time.time()
    best_route=func(*args)
    end_time=time.time()
    current,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    best_distance=total_distance(best_route,distance_lookup)
    execution_time=end_time-start_time
    return file_name,func_name,best_route,best_distance,execution_time,peak  
  
"""Parameters for all algorithms"""

# possible filenames: 'test.txt', 'bays29_1.txt','eil51_2.txt','berlin52.txt',
#                   'kroA100_4.txt','pr107_6.txt','gr120_3.txt','pa561_5.txt']

#list of files to run all algorithms listed in funcs_for_big_sets on
file_names_list = ['test.txt',
                   'bays29_1.txt',
                   'eil51_2.txt',
                   'berlin52.txt',
                   'kroA100_4.txt',
                   'pr107_6.txt',
                   'gr120_3.txt',
                   'pa561_5.txt']

#list of files to run exhaustive algorithm on
files_to_run_exhaustive=[]

#parameters for genetic algorithm
pop_size = 100    
tournament_size = 3
crossover_prob = 0.85
inversion_prob = 0.15
exchange_prob = 0.15
num_generations = 100000
elitism_ratio = 0.05

#list of functions to run on file_names_list
funcs_for_big_sets=[nearest_neighbor, mst_tsp, two_opt, NN_two_opt, MST_two_opt]

"""driver code to run the given algorithms on the given test sets"""                 
#create an array to store the output of each algorithm
array_length=(len(funcs_for_big_sets)+1)*len(files_to_run_exhaustive)+(len(file_names_list)-len(files_to_run_exhaustive))*len(funcs_for_big_sets)

dtype=[('City_set','U10'),
       ('Algorithm','U30'),
       ('Best Route', object),
       ('Best Distance',np.int32),
       ('Execution Time',np.float32),
       ('Peak Memory', np.int32)]

results_array=np.empty((array_length,),dtype=dtype)

#iterate through the test sets
i=0
for file_name in file_names_list:
    #load and create distance matrix and distance lookup
    triangular_matrix = load_triangular_matrix(file_name)
    symmetric_matrix = make_symmetric(triangular_matrix)
    distance_lookup = create_distance_lookup(symmetric_matrix)
    distance_matrix = np.array(symmetric_matrix)
    
    #initial route for two-opt, just the cities in numeric order
    two_opt_route=list(range(distance_matrix.shape[0]))
    two_opt_route.append(0)

    #dictionary of function names and the arguments necessary to run each
    func_dict={exhaustive_tsp:['exhaustive',[distance_matrix,distance_lookup]],
        nearest_neighbor:['nearest neighbor',[distance_matrix]],
           mst_tsp:['minimum spanning tree',[distance_matrix,distance_lookup]],
           genetic_algorithm_with_elitism:['genetic',[symmetric_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, elitism_ratio, distance_lookup]],
           two_opt:['two-opt',[two_opt_route,distance_lookup]],
           NN_two_opt:['nearest neighbor with 2-opt',[distance_matrix,distance_lookup]],
           MST_two_opt:['MST with 2-opt',[distance_matrix,distance_lookup]]}
    
    #iterate through the algorithms 
    for func, [func_name,[*args]] in func_dict.items():
        if file_name in files_to_run_exhaustive:
            results_array[i]=run_and_store_results(func, func_name, file_name,distance_lookup,*args)
            i+=1
        elif func in funcs_for_big_sets:
                results_array[i]=run_and_store_results(func, func_name, file_name,distance_lookup,*args)
                i+=1
                
#store results in CSV for analysis          
results_list=[{col: val for col, val in zip(results_array.dtype.names, row)} for row in results_array]
results_df=pd.DataFrame(results_list)
results_df['Best Route']=results_df['Best Route'].apply(lambda x:str(x))
results_df.to_csv('TSP_Results.csv',index=False)
