import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import numpy as np
from celluloid import Camera
import copy

SEED = 205
X, Y = 200, 200
OFFSET = .02
MAX_ELEV = .5
MIN_ELEV = -.5
MAX_STEEPNESS = .015 #/ ((X / 100)**2)# the max change in elevation between nodes that rover can travel between
START_COORD = (100, 165)
DESTINATION_COORD = (190, 72)
map = []
path_map = []
nodes = {}

fig = plt.figure()
camera = Camera(fig)
noise1 = PerlinNoise(octaves=3, seed=SEED)
noise2 = PerlinNoise(octaves=6, seed=SEED)
noise3 = PerlinNoise(octaves=12, seed=SEED)
noise4 = PerlinNoise(octaves=24, seed=SEED)

class Node:
    def __init__(self, coord):
        self.coord = coord
        self.parent = None
        self.visited = False
        self.distance = None
    
    def __repr__(self):
        return f"Node({self.coord})"

# changes the color of a square on the given map to value of 1 in cmap
def mark_square(coord, map):
    map[coord[1]][coord[0]] = 1

for i in range(X):
    row = []
    path_row = []
    for j in range(Y):
        noise_val = noise1([i/X, j/Y])
        noise_val += 0.5 * (noise2([i/X, j/Y]))
        noise_val += 0.25 * (noise3([i/X, j/Y]))
        noise_val += 0.125 * (noise4([i/X, j/Y]))

        row.append(noise_val)
        path_row.append(0)

    path_map.append(path_row)
    map.append(row)
'''
def directions(coord, nodes):
    x, y = coord
    can_up = y > 0
    can_left = x > 0
    can_right = x < len(no)
'''
#print(map)
diag_distance = 2 ** .5
# construct graph: populate the nodes list where (row,col) is index and value is list of edges
# adj_nodes list in the form [
#   coord: [(coord, distance), (coord, distance)],
#   coord2: etc.
# ]
for y, row in enumerate(map):
    for x, cur_val in enumerate(row):
        coord = (x, y)
        if coord not in nodes: nodes[coord] = []
        # REPLACE THIS WITH FUNCTION CALLS
        if y > 0 and abs(map[y-1][x] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x,y-1), 1)) # up
        if y < len(map)-1 and abs(map[y+1][x] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x,y+1), 1)) # down
        if x > 0 and abs(map[y][x-1] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x-1,y), 1)) # left
        if x < len(map[0])-1 and abs(map[y][x+1] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x+1,y), 1)) # right
        if y > 0 and x > 0 and abs(map[y-1][x-1] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x-1,y-1), diag_distance)) # up left
        if y > 0 and x < len(map[0])-1 and abs(map[y-1][x+1] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x+1,y-1), diag_distance)) # up right
        if y < len(map)-1 and x > 0 and abs(map[y+1][x-1] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x-1,y+1), diag_distance)) # down left
        if y < len(map)-1 and x < len(map[0])-1 and abs(map[y+1][x+1] - cur_val) <= MAX_STEEPNESS: nodes[coord].append(((x+1,y+1), diag_distance)) # down right

def bfs_search(adj_list, root, target):
    visited = [root]
    queue = [(root, [root])]
    while queue:
        node, path = queue.pop(0)
        adj_nodes = [x[0] for x in adj_list[node]] # access the node coords of all adj nodes
        for adj_node in adj_nodes:
            #adj_node = adj_node[0]
            if adj_node == target:
                return path + [adj_node]
            if adj_node not in visited:
                visited.append(adj_node)
                queue.append((adj_node, path + [adj_node]))
    
    return []

# it could be helpful to use djikstras if we want to change the weightings of edges (E.g Increase weight for large change in elevation, decrease weight for no elevation change)
def djikstra_search(adj_list, root, target):
    root_node = Node(root)
    root_node.distance = 0
    root_node.visited = True
    parents = {root: None} # key is node coord and value is parent coord
    node_dict = {root: root_node}
    prio_queue = [(root, est_func(root))] # frontier of nodes to be traversed to in order by distance

    # LOOP THROUGH PRIO QUEUE (FRONTIER)
    while prio_queue:
        # PARENT NODE
        parent_coord, parent_dist = prio_queue.pop(0)
        if parent_coord == target:
            found = True
            break
        adj_nodes = adj_list[parent_coord] # access the node coords of all adj nodes

        # CHILD NODES
        for adj_coord, dist in adj_nodes:
            if not adj_coord in node_dict: node_dict[adj_coord] = Node(adj_coord)
            adj_node = node_dict[adj_coord]

            if adj_node.visited: continue
            adj_node.visited = True
            new_dist = parent_dist + dist

            # Check if this new path reaches adj_node quicker than its current path
            if adj_node.distance and new_dist < adj_node.distance:
                parents[adj_coord] = parent_coord
                adj_node.distance = new_dist
                prio_queue.replace((adj_coord, dist), (adj_coord, new_dist))

            # If the adj node is unvisited then we add it to the prio queue
            else:
                parents[adj_coord] = parent_coord
                adj_node.distance = new_dist
                prio_queue.append((adj_coord, new_dist))

        prio_queue.sort(key=lambda x: x[1])
    
    # Extract the path from the dictionary that stores the parents of nodes
    if found:
        temp = []
        while target != None:
            temp.append(target)
            target = parents[target]
        temp.append(root)
        temp.reverse()
        return temp
    return None

# make list of coords path from adj list

map = np.array(map)
search_func = bfs_search
path = search_func(nodes, START_COORD, DESTINATION_COORD)
if path:
    print(f"A path with distance {len(path)} has been found")
    target_node = DESTINATION_COORD
else:
    print("No path found")

for coord in path:
    mark_square(coord, map)
    plt.imshow(path_map, alpha=.5, cmap='autumn')
    plt.imshow(map, cmap='gray')
    camera.snap()
#b = -.1
#contour_frame = np.array(pic)
#contour_frame[np.where((contour_frame < b-offset) | (contour_frame > b+offset))] = 0
#plt.imshow(contour_frame, cmap='gray')
#plt.show()
animation = camera.animate(interval=50)
animation.save('topo_path.gif')
plt.imshow(path_map, alpha=.5, cmap='autumn')
plt.imshow(map, cmap='gray')
plt.show()
