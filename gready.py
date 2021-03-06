#!usr/bin/env python3
from numpy import nextafter, random, result_type
from math import inf
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import trace
import heapq

FILENAME = './maze/maze_8.txt'

# Draw map
def visualize_maze(matrix, bonus, start, end, route=None):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')

        direction.pop(0)

    #2. Drawing the map
    ax=plt.figure(dpi=100,figsize=(10,10)).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    
    plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green')

    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],
                        marker=direction[i],color='#1dcc22')

    plt.text(end[1],-end[0],'EXIT',color='red',
         horizontalalignment='center',
         verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')
    
    for _, point in enumerate(bonus):
      print(f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')

#Read map
def read_file(file_name: str = 'maze.txt'):
  f=open(file_name,'r')
  n_bonus_points = int(next(f)[:-1])
  bonus_points = []
  for i in range(n_bonus_points):
    x, y, reward = map(int, next(f)[:-1].split(' '))
    bonus_points.append((x, y, reward))

  text=f.read()
  matrix=[list(i) for i in text.splitlines()]
  f.close()

  return bonus_points, matrix

class Node_dfs:
    def __init__(self, data):
        self.visited = False
        self.data = data
        self.parent = None
        self.g = inf
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
def distance_between_2_nodes(node_a, node_b):
    h = abs(node_a.data[0] - node_b.data[0]) + \
        abs(node_a.data[1] - node_b.data[1])
    return h
  
def greedy(matrix_node, bonus, current, end, result):
  if len(result) > 0:
    return
  if np.array_equal(current.data, end.data):
    node = current
    end.g = current.g
    while node:
      result.append(node.data)
      node = matrix_node[node.data[0]][node.data[1]].parent
    return
    
  queue = PriorityQueue()
  for i in range(-1, 2):
      for j in range(-1, 2):
        if abs(i) != abs(j):
          x = current.data[0]+i
          y = current.data[1]+j
          if (x < 0) or (y < 0) or (x >= len(matrix_node)) or (y >= len(matrix_node[0])) \
            or (matrix_node[x][y].visited):
            continue
          else:
           
            queue.push(Node_dfs([x,y]), matrix_node[x][y].g)
  while queue.size():
    next_node = queue.pop()
    next_node.parent = current
    next_node.g = current.g + bonus[next_node.data[0]][next_node.data[1]]
    matrix_node[next_node.data[0]][next_node.data[1]].visited = True
    matrix_node[next_node.data[0]][next_node.data[1]].parent = current
    greedy(matrix_node, bonus, next_node, end, result)
  
bonus_points, matrix = read_file(FILENAME)
matrix_node = []
start = []
end = []
bonus = []
for i in range(len(matrix)):
  matrix_node.append([])
  bonus.append([])
  for j in range(len(matrix[i])):
    if matrix[i][j] == 'x':
      node = Node_dfs([i,j])
      node.visited = True
      matrix_node[i].append(node)
      bonus[i].append(inf)
    elif matrix[i][j] == ' ':
      if i == 0 or i == len(matrix)-1 or j == 0 or j == (len(matrix[i])-1):
        end.extend([i, j])
      matrix_node[i].append(Node_dfs([i, j]))
      bonus[i].append(1)
    elif matrix[i][j] == 'S':
      start.extend([i, j])
      node = Node_dfs([i,j])
      node.visited = True
      matrix_node[i].append(node)
      bonus[i].append(0)
    else:
      matrix_node[i].append(Node_dfs([i, j]))
      bonus[i].append(0)


for point in bonus_points:
  bonus[point[0]][point[1]] = point[2]

end_node = Node_dfs(end)

for row in matrix_node:
  for node in row:
    node.g = distance_between_2_nodes(node, end_node) \
    + bonus[node.data[0]][node.data[1]]

start_node = Node_dfs(start)
start_node.visited = True
start_node.g = 0
result =[]
greedy(matrix_node, bonus, start_node, end_node, result)
print('Number of step:  ', end_node.g)
visualize_maze(matrix, bonus_points, start, end, result[::-1])
