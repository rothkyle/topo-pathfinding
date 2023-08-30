import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import numpy as np
plt.style.use('seaborn-pastel')

SEED = 202
X, Y = 100, 100
OFFSET = .03
MAX_ELEV = .5
MIN_ELEV = -.5
MAX_STEEPNESS = .1 # the max change in elevation between nodes that rover can travel between
map = []
nodes = []

noise1 = PerlinNoise(octaves=3, seed=SEED)
noise2 = PerlinNoise(octaves=6, seed=SEED)
noise3 = PerlinNoise(octaves=12, seed=SEED)
noise4 = PerlinNoise(octaves=24, seed=SEED)

"""
class Node:
    def __init__(self, row, col, elevation, adj_nodes=[]):
        self.row = row
        self.col = col
        self.elevation = elevation
        self.adj_nodes = adj_nodes

    def add_edge(self, edge):
        self.adj_nodes.append(edge)
"""

for i in range(X):
    row = []
    for j in range(Y):
        noise_val = noise1([i/X, j/Y])
        noise_val += 0.5 * (noise2([i/X, j/Y]))
        noise_val += 0.25 * (noise3([i/X, j/Y]))
        noise_val += 0.125 * (noise4([i/X, j/Y]))

        row.append(noise_val)
    map.append(row)

# construct graph: populate the nodes list where (row,col) is index and value is list of edges
for r, row in enumerate(map):
    for c, col in enumerate(row):
        coord = (row,col)
        cur_val = map[row][col]
        if coord not in nodes: nodes[coord] = []
        if r > 0 and abs(map[row - 1][col] - cur_val) <= MAX_STEEPNESS: map[coord].append((row-1,col)) # up
        if r < len(map) and abs(map[row + 1][col] - cur_val) <= MAX_STEEPNESS: map[coord].append((row+1,col)) # down
        if c > 0 and abs(map[row][col - 1] - cur_val) <= MAX_STEEPNESS: map[coord].append((row-1,col)) # left
        if c < len(map[0]) and abs(map[row][col + 1] - cur_val) <= MAX_STEEPNESS: map[coord].append((row+1,col)) # right

        # DOESNT CONSIDER DIAGONAL DISTANCE

print(max(map))
#b = -.1
#contour_frame = np.array(pic)
#contour_frame[np.where((contour_frame < b-offset) | (contour_frame > b+offset))] = 0
#plt.imshow(contour_frame, cmap='gray')
#plt.show()

plt.imshow(map, cmap='gray')
plt.show()