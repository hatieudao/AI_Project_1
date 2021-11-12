#!usr/bin/env python3
from numpy import nextafter, random, result_type
from math import inf
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import trace


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
    ax=plt.figure(dpi=100).add_subplot(111)

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

def dfs(matrix_node, bonus, current, end, trace, result):
  val = end.g
  if current.g >= val:
    return
  if np.array_equal(current.data, end.data):
    if val > current.g:
      end.g = current.g
      end.parent = current
      result.clear()
      result.extend(trace)
  next_node = Node_dfs([-1,-1])
  for i in range(-1, 2):
      for j in range(-1, 2):
        if abs(i) != abs(j):
          x = current.data[0]+i
          y = current.data[1]+j
          next_node = Node_dfs([x,y])
          if (x < 0) or (y < 0) or (x >= len(matrix_node)) or (y >= len(matrix_node[0])) or matrix_node[x][y].visited:
            continue
          else:
            matrix_node[x][y].parent = current
            next_node.g = current.g + bonus[x][y]
            matrix_node[x][y].g = next_node.g
            matrix_node[x][y].visited = True
            trace.append(current.data)
            dfs(matrix_node, bonus, next_node, end, trace, result)
            matrix_node[x][y].g = inf
            matrix_node[x][y].visited = False
            trace.pop(-1)
    
bonus_points, matrix = read_file('maze_map.txt')
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

start_node = Node_dfs(start)
start_node.visited = True
start_node.g = 0
end_node = Node_dfs(end)
result =[start]
trace = []
dfs(matrix_node, bonus, start_node, end_node, trace, result)
route = result.append(end)
print('Number of step:  ', end_node.g)
visualize_maze(matrix, bonus_points, start, end, result)
