#!usr/bin/env python3
from numpy import random

W_RATIO = 0.25
W_RATIO2 = 0.2
def write_map(name, n, m, maze, bonus_points):
  with open(name, 'w') as outfile:
    outfile.write(f'{len(bonus_points)}\n')
    for point in bonus_points:
      outfile.write(f'{point[0]} {point[1]} {point[2]} \n')
    for line in maze:
      outfile.write(line+'\n')
def gen_data(pos, n, m, b):
  #generate bonus point
  bonus_points = []
  for i in range(b):
    isContinue = True
    while isContinue: #check existed point
      x = random.randint(n-1)
      y = random.randint(m-1)
      isContinue = False
      for point in bonus_points:
        if x == point[0] and y == point[1]:
          isContinue = True
          break
      if isContinue == False:
        bonus_points.append([x, y, - random.randint(5)])
  maze = []
  exit = random.randint(n)
  # Draw map
  for i in range(n):
    s = ''
    nw = int(m*W_RATIO)
    for j in range(m):
      if i == 0 or i==n-1 or (i!=exit and j == 0) or j == m - 1:
        s += 'x'
      elif i == exit and j < 3:
        s += ' '
      else:
        t = random.randint(2)
        nw -= t
        if nw <= 0:
          t = 0
          nw += random.randint(int(m*W_RATIO2))
        s +=  t*'x' + (1-t)*' '
    maze.append(s)  
  
  for point in bonus_points:
    maze[point[0]][point[1]] = '+'

  # Start point
  isContinue = True
  while isContinue: #check existed point
    x = random.randint(n-1)
    y = random.randint(m-1)
    isContinue = False
    for point in bonus_points:
      if (x == point[0] and y == point[1]) or (x == 0 or x == n-1 or y == 0 or y == m-1):
        isContinue = True
        break
    if isContinue == False:
        s = maze[x]
        s = list(s)
        s[y] = 'S'
        maze[x] = ''.join(s)
  
  write_map(f'maze_{pos}.txt', n, m, maze, bonus_points)
  
for i in range(3):
  gen_data(i, 35, 15, 0)

