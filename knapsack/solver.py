#!/usr/bin/python
# -*- coding: utf-8 -*-

#from collections import namedtuple
#Item = namedtuple("Item", ['index', 'value', 'weight', 'select'])

import sys

class Item:
    def __init__(self, index, value, weight, select):
        self.index = index
        self.value = value
        self.weight = weight
        self.select = select
        self.ratio = 1

    def __repr__(self):
        return '{index: %s, value: %s, weight: %s, ratio: %s, select: %s}' % (self.index, self.value, self.weight, self.ratio, self.select)

def backtrace_dp(m, k, n, items):

    #print("Debug messages...{} {}".format(k, n), file=sys.stderr)

    if k == 0 or n == 0:
        return

    if m[k][n] == m[k][n-1]:
        items[n-1].select = 0
        backtrace_dp(m, k, n-1, items)
    else:
        items[n - 1].select = 1
        backtrace_dp(m, k-items[n-1].weight, n - 1, items)

def calc_estimate(items, capacity, idx, max_tree_depth):
    weight = 0
    value = 0
    for elem in range(idx, max_tree_depth+1):
        if weight + items[elem].weight <= capacity:
            weight += items[elem].weight
            value += items[elem].value
        else:
            return value + items[elem].ratio * (capacity - weight)
    return value

def knapsack_DP(items, K):

    '''
    # include 0 capacity
    m = [0] * (capacity+1)
    # include item 0
    num_items = len(items)+1
    for c in range(capacity+1):
        m[c] = [0] * num_items
    '''

    n = len(items)
    m = [[0 for x in range(n+1)] for y in range(K+1)]

    for item in items:
        for k in range(1, K+1):
            # can't take the current item for capacity k
            if item.weight > k:
                m[k][item.index] = m[k][item.index-1]
            else:
                # check if taking curr. item will give u a higher value
                m[k][item.index] = max(m[k][item.index-1], m[k-item.weight][item.index-1] + item.value)

    #print("Debug messages...{} {}".format(len(m), len(m[0])), file=sys.stderr)
    backtrace_dp(m, K, n, items)

    output_data = str(m[K][n]) + ' ' +  str(1) + '\n'
    taken = []
    #print("Debug messages...len of items: {}".format(len(items)), file=sys.stderr)

    for item in items:
        taken.append(item.select)

    #print("Debug messages...{} {}".format(len(taken), taken), file=sys.stderr)
    output_data += ' '.join(map(str, taken))
    return output_data

def trivial_greedy(items, capacity):
    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0] * len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def greedy_algorithm_ver2(items,capacity):
    taken = [0]*len(items)
    order = [x for x in range(len(items))]
    order = sorted(order, key=lambda x: -float(items[x].value) / items[x].weight)
    weight = 0
    value = 0
    for elem in order:
        if weight + items[elem].weight <= capacity:
            weight += items[elem].weight
            value += items[elem].value
            taken[elem] = 1
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

class Node:
    def __init__(self, index, value, room, estimate, path):
        self.index = index
        self.value = value
        self.room = room
        self.estimate = estimate
        self.path =  path

    def __repr__(self):
        return 'index: %s, value: %s, room: %s, estimate: %s' % (self.index, self.value, self.room, self.estimate)

def child_node(node, items, taken, best_node, max_tree_depth):
    item = items[node.index + 1]
    path = node.path
    path += " %d" % taken

    child = None

    if taken:
        if item.weight <= node.room:
            child = Node(node.index+1, node.value + item.value, node.room - item.weight, node.estimate, path)
    else:
        new_estimate = node.value + calc_estimate(items, node.room, node.index+2, max_tree_depth)

        # pruning
        if best_node and new_estimate <= best_node.estimate:
            #print('pruning at node %s' % str(node.index+1))
            return None

        child = Node(node.index+1, node.value, node.room, new_estimate, path)

    return child

def branch_and_bound(items, capacity):

    # calc value per weight
    for item in items:
        item.ratio = item.value / item.weight

    # sort items
    sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)

    max_tree_depth = len(items)
    estimate = calc_estimate(sorted_items, capacity, 0, max_tree_depth) #sum(item.value for item in items)
    best_node = Node(0, 0, capacity, 0, [])

    # for easier index only
    sorted_items.insert(0, 'dummy')
    stack = []

    node_0 = Node(0, 0, capacity, estimate, '')
    stack.append(node_0)

    while(len(stack)):
        node = stack.pop()

        # leaf node
        if node.index == max_tree_depth:
            # update best_node
            if node.estimate > best_node.estimate:
                best_node = node
        else:
            # does the node has valid children ?
            for taken in [0, 1]:
                child = child_node(node, sorted_items, taken, best_node, max_tree_depth)

                if child:
                    stack.append(child)

    sorted_items.pop(0)
    best_path = best_node.path.split(' ')
    for i, item in enumerate(sorted_items):
        item.select = best_path[i]

    #restore result order
    sorted_items = sorted(items, key=lambda x: x.index)
    path = []
    for item in sorted_items:
        path.append(item.select)

    output_data = str(best_node.value) + ' ' +  str(1) + '\n'
    output_data += ' '.join(map(str, path))
    return output_data

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    print("Debug messages...{}".format(firstLine), file=sys.stderr)

    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i, int(parts[0]), int(parts[1]), 0))

    #print("Debug messages...{}".format(items), file=sys.stderr)

    if len(items) <= 200:
        return knapsack_DP(items, capacity)
    elif len(items):
        return branch_and_bound(items, capacity)
    #else:
    #    return greedy_algorithm_ver2(items, capacity)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

