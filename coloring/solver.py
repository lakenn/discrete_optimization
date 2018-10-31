#!/usr/bin/python
# -*- coding: utf-8 -*-

from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

class Node:
    def __init__(self, index):
        self.index = index
        self.degree = 0
        self.neighbours = set()
        self.color = -1
        self.dv = []

    def __repr__(self):
        return 'index: %s, degree: %d, neighbours: %s, color: %d' % (self.index, self.degree, self.neighbours, self.color)

def cp_sat_solver(nodes, edges, greedy_count, node_count, edge_count):
    # sort the nodes in DESC degrees
    sorted_nodes = sorted(nodes.values(), key=lambda x: x.degree, reverse=True)

    print([node.index for node in sorted_nodes])

    model = cp_model.CpModel()

    # x[i,c] = 1 means that node i is assigned color c
    x = {}
    all_decision_nodes = []
    for i, node in enumerate(sorted_nodes):
        v = node.index
        for j in range(greedy_count):
            dv = model.NewIntVar(0, 1, 'v[%i,%i]' % (v, j))
            x[v, j] = dv
            all_decision_nodes.append(dv)
            node.dv.append(dv)

            # assign lower possible color to x first
            if j > i:
                model.Add(x[v,j] == 0)

    # u[c] = 1 means that color c is used, i.e. assigned to some node
    u = [model.NewIntVar(0, 1, 'u[%i]' % i) for i in range(greedy_count)]

    #
    # constraints
    #

    # each node must be assigned exactly one color
    for i in range(node_count):
        #model.AddSumConstraint([x[i, c] for c in range(greedy_count)], 1, 1)
        model.Add(sum([x[i, c] for c in range(greedy_count)]) == 1)

    # adjacent nodes cannot be assigned the same color
    for i in range(edge_count):
        for c in range(greedy_count):
            model.Add(x[edges[i][0], c] + x[edges[i][1], c] <= u[c])
            #model.Add(x[edges[i][0], c] != x[edges[i][1], c]).OnlyEnforceIf(u[c])

    # force first node color = 0
    first_node = sorted_nodes[0]
    model.Add(first_node.dv[0] == 1)
    model.Add(u[0] == 1)

    # node i is assigned color k only when color k is active
    #for constraint in (x[i, k] - u[k] <= 0 for i in range(node_count) for k in range(greedy_count)):
    #    model.Add(constraint)
    #for i in range(node_count):
    #    for k in range(greedy_count):
    #        model.Add(u[k] == 1).OnlyEnforceIf(x[i,k])

    # symmetry breaking
    # color index should be as low as possible
    for constraint in (u[i] - u[i + 1] >= 0 for i in range(greedy_count - 1)):
        model.Add(constraint)

    # Minimize number of colors used
    model.Minimize(sum(u))

    # Search for x values in increasing order.
    model.AddDecisionStrategy(all_decision_nodes, cp_model.CHOOSE_FIRST,
                              cp_model.SELECT_MAX_VALUE)


    solver = cp_model.CpSolver()
    # Force solver to follow the decision strategy exactly.
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    #solver.parameters.num_search_workers = 4

    # Sets a time limit of 120 seconds.
    solver.parameters.max_time_in_seconds = 480

    status = solver.Solve(model)

    print('Solve status: %s' % solver.StatusName(status))
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Optimal objective value: %i' % solver.ObjectiveValue())

        print('Statistics')
        print('  - conflicts : %i' % solver.NumConflicts())
        print('  - branches  : %i' % solver.NumBranches())
        print('  - wall time : %f s' % solver.WallTime())


        for node in sorted_nodes:
            for j, dv in enumerate(node.dv):
                if solver.Value(dv):
                    node.color = j
                    break

        # restore result order
        original_nodes = sorted(sorted_nodes, key=lambda x: x.index)
        results = []
        for node in original_nodes:
            results.append(node.color)

        return status, int(solver.ObjectiveValue()), results

    return 0, 0, []

def linear_solver(nodes, edges, greedy_count, node_count, edge_count):

    # use result of greedy to limit search space

    solver = pywraplp.Solver('CoinsGridCLP',
                            pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # x[i,c] = 1 means that node i is assigned color c
    x = {}
    for v in range(node_count):
        for j in range(greedy_count):
            x[v, j] = solver.IntVar(0, 1, 'v[%i,%i]' % (v, j))


    # u[c] = 1 means that color c is used, i.e. assigned to some node
    u = [solver.IntVar(0, 1, 'u[%i]' % i) for i in range(greedy_count)]

    # number of colors used, to minimize
    obj = solver.Sum(u)

    #
    # constraints
    #

    # each node must be assigned exactly one color
    for i in range(node_count):
        solver.Add(solver.Sum([x[i, c] for c in range(greedy_count)]) == 1)

    # adjacent nodes cannot be assigned the same color
    # (and adjust to 0-based)
    for i in range(edge_count):
        for c in range(greedy_count):
            solver.Add(x[edges[i][0], c] + x[edges[i][1], c] <= u[c])

    # force first node color = 0
    solver.Add(x[0, 0] == 1)

    # objective
    solver.Minimize(obj)

    #
    # solution
    #
    solver.Solve()
    print()
    print('number of colors:', int(solver.Objective().Value()))
    print('colors used:', [int(u[i].SolutionValue()) for i in range(greedy_count)])
    print()

    for v in range(node_count):
        print('v%i' % v, ' color ', end=' ')
        for c in range(greedy_count):
          if int(x[v, c].SolutionValue()) == 1:
            print(c)

def greedy_color(nodes):

    all_colors = set(range(-1, len(nodes)))

    # sort the nodes in DESC degrees
    sorted_nodes = sorted(nodes.values(), key=lambda x: x.degree, reverse=True)

    for node in sorted_nodes:

        used_colors = set([-1])
        # check adj nodes' color
        for adj_node in node.neighbours:
            used_colors.add(nodes[adj_node].color)

        # get the min color of available color
        available_color = all_colors - used_colors
        nodes[node.index].color = min(available_color)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    print('Node: %d, Edge: %d' % (node_count, edge_count))
    # ken's code
    nodes = {k: Node(k) for k in range(node_count)}

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

        # ken's code
        n1, n2 = int(parts[0]), int(parts[1])
        nodes[n1].neighbours.add(n2)
        nodes[n2].neighbours.add(n1)

        nodes[n1].degree += 1
        nodes[n2].degree += 1

    # greedy approach
    greedy_color(nodes)

    max_color = 0
    greedy_results = []
    for node in nodes.values():
        if node.color > max_color:
            max_color = node.color

        greedy_results.append(node.color)
    max_color += 1

    results = []

    if node_count <= 500:
        # solver approach
        status, solution, results = cp_sat_solver(nodes, edges, max_color, node_count, edge_count)
        #linear_solver(nodes, edges, max_color, node_count, edge_count)

    if len(results) and solution < max_color:
        if status == cp_model.OPTIMAL:
            status = 1
        else:
            status = 0

        output_data = str(solution) + ' ' + str(status) + '\n'

        output_data += ' '.join(map(str, results))
        return output_data

    else:
        output_data = str(max_color) + ' ' + str(0) + '\n'

        output_data += ' '.join(map(str, greedy_results))
        return output_data

    '''
    # build a trivial solution
    # every node has its own color
    solution = range(0, node_count)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data
    '''


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

