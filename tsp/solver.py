#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from random import sample

from collections import namedtuple
from sklearn.neighbors import KDTree

Point = namedtuple("Point", ['x', 'y', 'index'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def manhattan_length(point1, point2):
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)

def kd_nn(cities):
    points = [(city.x, city.y) for city in cities]
    tree = KDTree(points, leaf_size=10, metric='euclidean')
    results = tree.query(points, k=2, return_distance=False)
    return results

def compute_total_manthan_distance(solution):

    nodeCount = len(solution)
    # calculate the length of the tour
    obj = manhattan_length(solution[-1], solution[0])
    for index in range(0, nodeCount-1):
        obj += manhattan_length(solution[index], solution[index + 1])

    return obj

def nearest_neighbour(cities):

    # sort all cities according to x
    cities = sorted(cities, key = lambda city: city.x)

    current_city = cities[0]
    cloest_city = None

    path = [cities.pop()]

    while(len(cities)):
        min_dist = float("inf")

        for city in cities:
            dist = manhattan_length(current_city, city)

            if dist < min_dist:
                min_dist = dist
                cloest_city = city

        path.append(cloest_city)
        cities.remove(cloest_city)
        current_city = cloest_city

    return path

# Pairwise exchange (2-opt)
def generate_solution(cities):
    return sample(cities, len(cities))

# swap two edges
def swap(solution, x, y):
    return solution[:x] + solution[x:y + 1][::-1] + solution[y + 1:]

def pairwise_exchange(points):
    solution = generate_solution(points)
    stable, best = False, compute_total_manthan_distance(solution)

    while not stable:
        stable = True
        for i in range(1, len(solution) - 1):
            for j in range(i + 1, len(solution)):
                candidate = swap(solution, i, j)
                length_candidate = compute_total_manthan_distance(candidate)

                if best > length_candidate:
                    solution, best = candidate, length_candidate
                    stable = False

    return solution

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    print(nodeCount)

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), i-1))

    '''
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    print(nodeCount)
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    solution = nearest_neighbour(points)
    # calculate the length of the tour
    obj = length(solution[-1], solution[0])
    for index in range(0, nodeCount-1):
        obj += length(solution[index], solution[index+1])


    print(obj)
    '''

    if len(points) < 600:
        solution = pairwise_exchange(points)
    else:
        solution = nearest_neighbour(points)

    # calculate the length of the tour
    obj = length(solution[-1], solution[0])
    for index in range(0, nodeCount-1):
        obj += length(solution[index], solution[index+1])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, [ sol.index for sol in solution ]))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

