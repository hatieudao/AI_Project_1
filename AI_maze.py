from numpy import random
from math import inf
import heapq
import os
import matplotlib.pyplot as plt

FILENAME = '../maze/maze_8.txt'
WALL = inf
SIZE = (11, 5)


class PriorityQueue:
    def __init__(self):
        self._data = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._data, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._data)[-1]

    def size(self):
        return len(self._data)


class Node_bfs:
    def __init__(self, data):
        self.visited = False
        self.data = data
        self.parent = None
        self.g = inf


class Node_dfs:
    def __init__(self, data):
        self.visited = False
        self.data = data
        self.parent = None
        self.g = inf


class Node_greedy:
    def __init__(self, data, end):
        self.data = data
        self.visited = False
        self.g = inf
        self.h = distance_between_2_nodes(self.data, end)
        self.parent = None


class Node_a_star:
    def __init__(self, data, end):
        self.data = data
        self.visited = False
        self.g = inf
        self.h = distance_between_2_nodes(self.data, end)
        self.parent = None

    def calculate_f(self):
        return self.g+self.h


def visualize_maze(matrix, bonus, start, end, size, route=None):
    walls = [(i, j) for i in range(len(matrix))
             for j in range(len(matrix[0])) if matrix[i][j] == 'x']
    if route:
        direction = []
        for i in range(1, len(route)):
            if route[i][0]-route[i-1][0] > 0:
                direction.append('^')  # ^
            elif route[i][0]-route[i-1][0] < 0:
                direction.append('v')  # v
            elif route[i][1]-route[i-1][1] > 0:
                direction.append('<')
            else:
                direction.append('>')

        direction.pop(0)

    # 2. Drawing the map
    ax = plt.figure(dpi=100, figsize=size).add_subplot(111)

    for i in ['top', 'bottom', 'right', 'left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls], [-i[0] for i in walls],
                marker='X', s=100, color='black')

    plt.scatter([i[1] for i in bonus], [-i[0] for i in bonus],
                marker='P', s=100, color='green')

    plt.scatter(start[1], -start[0], marker='*',
                s=100, color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1], -route[i+1][0],
                        marker=direction[i], color='red')

    plt.text(end[1], -end[0], 'EXIT', color='red',
             horizontalalignment='center',
             verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')

    for _, point in enumerate(bonus):
        print(
            f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')


def read_file(file_name):
    f = open(file_name, 'r')
    n_bonus_points = int(next(f)[:-1])
    bonus_points = []
    for i in range(n_bonus_points):
        bonus_point = list(map(int, next(f)[:-1].split(' ')))
        bonus_points.append(bonus_point)
    text = f.read()
    matrix = [list(i) for i in text.splitlines()]
    f.close()
    return bonus_points, matrix


def weight(maze, node):
    return maze[node.data[0]][node.data[1]]


def distance_between_2_nodes(node_data_a, node__data_b):
    h = abs(node_data_a[0] - node__data_b[0]) + \
        abs(node_data_a[1] - node__data_b[1])
    return h


def back_track_node(node_end):
    back_track_node = node_end
    route = []
    while back_track_node:
        route.append(back_track_node.data)
        back_track_node = back_track_node.parent
    return route


def bfs(matrix_node, maze, start, end, result):
    matrix_node[start[0]][start[1]].visited = True
    matrix_node[start[0]][start[1]].g = 0
    queue = []
    queue.append(matrix_node[start[0]][start[1]])
    while queue:
        node = queue.pop(0)
        if node.data == end:
            result = back_track_node(node)
            return result
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) != abs(j):
                    x = node.data[0]+i
                    y = node.data[1]+j
                    next_node = matrix_node[x][y]
                    if maze[x][y] != WALL and next_node.visited == False:
                        next_node.g = node.g+weight(maze, next_node)
                        next_node.parent = node
                        next_node.visited = True
                        queue.append(next_node)


def dfs(matrix_node, maze, current, end, result):
    if len(result) > 0:
        return result
    node = matrix_node[current[0]][current[1]]
    if current == end:
        result = back_track_node(node)
        return result
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i) != abs(j):
                x = current[0]+i
                y = current[1]+j
                next_node = matrix_node[x][y]
                if maze[x][y] != WALL and matrix_node[x][y].visited == False:
                    next_node.parent = node
                    next_node.g = node.g + weight(maze, next_node)
                    next_node.visited = True
                    result_route = dfs(matrix_node, maze, [x, y], end, result)
                    if result_route:
                        if len(result_route) > 0:
                            return result_route


def greedy(matrix_node, maze, start, end, result):
    matrix_node[start[0]][start[1]].visited = True
    matrix_node[start[0]][start[1]].g = 0
    queue = PriorityQueue()
    queue.push(matrix_node[start[0]][start[1]],
               matrix_node[start[0]][start[1]].h+maze[start[0]][start[1]])
    while queue.size() > 0:
        node = queue.pop()
        if(node.data == end):
            result = back_track_node(node)
            return result
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) != abs(j):
                    x = node.data[0]+i
                    y = node.data[1]+j
                    next_node = matrix_node[x][y]
                    if maze[x][y] != WALL and next_node.visited == False:
                        next_node.g = node.g+weight(maze, next_node)
                        next_node.parent = node
                        next_node.visited = True
                        queue.push(next_node, next_node.h +
                                   weight(maze, next_node))
                        maze[x][y] = 1


def a_star(matrix_node, maze, start, end, result):
    matrix_node[start[0]][start[1]].visited = True
    matrix_node[start[0]][start[1]].g = 0
    queue = PriorityQueue()
    queue.push(matrix_node[start[0]][start[1]],
               matrix_node[start[0]][start[1]].calculate_f())
    while queue.size() > 0:
        node = queue.pop()
        if(node.data == end):
            result = back_track_node(node)
            return result
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) != abs(j):
                    x = node.data[0]+i
                    y = node.data[1]+j
                    next_node = matrix_node[x][y]
                    if maze[x][y] != WALL:
                        if next_node.visited == True:
                            new_f = node.g + next_node.h + \
                                weight(maze, next_node)
                            if new_f < next_node.calculate_f():
                                next_node = Node_a_star([x, y], end)
                                next_node.g = node.g + \
                                    weight(maze, next_node)
                                next_node.visited = True
                                next_node.parent = node
                                queue.push(next_node,
                                           next_node.calculate_f())
                        else:
                            next_node.g = node.g + \
                                weight(maze, next_node)
                            next_node.parent = node
                            next_node.visited = True
                            queue.push(next_node,
                                       next_node.calculate_f())
                            # ăn điểm thưởng rồi thì node này sẽ có giá trị là 1
                            maze[x][y] = 1


def get_matrix_weight_start_point_end_point(bonus_points, matrix):
    maze = []
    start = []
    end = []
    for i in range(len(matrix)):
        maze.append([])
        for j in range(len(matrix[i])):
            if matrix[i][j] == 'x':
                maze[i].append(WALL)
            elif matrix[i][j] == ' ':
                if i == 0 or i == len(matrix)-1 or j == 0 or j == (len(matrix[i])-1):
                    end.extend([i, j])
                maze[i].append(1)
            elif matrix[i][j] == '+':
                for bonus_point in bonus_points:
                    if i == bonus_point[0] and j == bonus_point[1]:
                        maze[i].append(bonus_point[2])
            elif matrix[i][j] == 'S':
                maze[i].append(9)
                start.extend([i, j])
    return maze, start, end


def get_matrix_node(maze, end, algorithm):
    matrix_node = []
    for i in range(len(maze)):
        matrix_node.append([])
        for j in range(len(maze[i])):
            if algorithm == 'bfs':
                matrix_node[i].append(Node_bfs([i, j]))
            elif algorithm == 'dfs':
                matrix_node[i].append(Node_dfs([i, j]))
            elif algorithm == 'greedy':
                matrix_node[i].append(Node_greedy([i, j], end))
            else:
                matrix_node[i].append(Node_a_star([i, j], end))
    return matrix_node


def run_algorithm(algorithm, FILENAME, SIZE):
    bonus_points, matrix = read_file(FILENAME)
    maze, start, end = get_matrix_weight_start_point_end_point(
        bonus_points, matrix)
    matrix_node = get_matrix_node(maze, end, algorithm)
    matrix_node[start[0]][start[1]].visited = True
    matrix_node[start[0]][start[1]].g = 0
    if algorithm == 'bfs':
        route = bfs(matrix_node, maze, start, end, [])
    elif algorithm == 'dfs':
        route = dfs(matrix_node, maze, start, end, [])
    elif algorithm == 'greedy':
        route = greedy(matrix_node, maze, start, end, [])
    elif algorithm == 'a_star':
        route = a_star(matrix_node, maze, start, end, [])
    print('chi phí thực hiện:', matrix_node[end[0]][end[1]].g)
    if route:
        print(route)
        visualize_maze(matrix, bonus_points, start, end, SIZE, route)
    else:
        print('không tìm thấy đường đi thoát khỏi mê cung')


# run BFS
run_algorithm('bfs', FILENAME, SIZE)
# # run DFS
run_algorithm('dfs', FILENAME, SIZE)
# # run greedy
run_algorithm('greedy', FILENAME, SIZE)
# run a start
run_algorithm('a_star', FILENAME, SIZE)
