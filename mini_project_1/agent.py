import random
import matplotlib.pyplot as plt  # Add matplotlib for plotting
from turtle import st #imported random library for grid creation and dirt randomness

def make_random_grid(rows, cols, dirt_prob): #function for grid creation
    return [[1 if random.random() < dirt_prob else 0 for _ in range(cols)] for _ in range(rows)] #randomly puts dirt in cells

def print_grid(grid, agent_pos): #makes the visual grid to track movement
    for x in range(len(grid)): 
        for y in range(len(grid[0])):
            if (x, y) == agent_pos:
                print("[A]", end=" ") #finds agent position and signifies it
            else:
                print(f"[{grid[x][y]}]", end=" ") #takes inputs and adds them to grid creation
        print()
    print()

def bfs(start, targets, m, n):
    queue = [(start, [])]
    visited = set([start])
    maxq = 0
    while queue:
        (x, y), path = queue.pop(0)
        if (x, y) in targets:
            return path + [(x, y)], maxq  # Return both path and maxq
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(x, y)]))
                if len(queue) > maxq:
                    maxq = len(queue)
    return None, maxq  # Return maxq even if no path found

def vacuum_world_bfs(m, n):
    dirt_prob = 0.5
    grid = make_random_grid(m, n, dirt_prob)
    agent_pos = (0, 0)
    steps = 0
    cleans = 0
    maxq_overall = 0

    while any(1 in row for row in grid):
        dirty_cells = {(x, y) for x in range(m) for y in range(n) if grid[x][y] == 1}
        path, maxq = bfs(agent_pos, dirty_cells, m, n)
        if maxq > maxq_overall:
            maxq_overall = maxq
        for pos in path:
            agent_pos = pos
            steps += 1
            x, y = agent_pos
            if grid[x][y] == 1:
                grid[x][y] = 0
                cleans += 1
    return steps, maxq_overall  # Return steps instead of ratio

initial_m = 10
initial_n = 10

steps_data = []
queue_data = []
grid_labels = []

for i in range(5):
    m = initial_m + i * 10
    n = initial_n + i * 10
    print(f"Grid Size: {m}x{n}")
    steps_runs = []
    queue_runs = []
    for j in range(10):
        steps, maxq = vacuum_world_bfs(m, n)
        steps_runs.append(steps)
        queue_runs.append(maxq)
        print(f"  Run {j+1}: Steps: {steps}, Max Queue Size: {maxq}")
    steps_data.append(steps_runs)
    queue_data.append(queue_runs)
    grid_labels.append(f"{m}x{n}")

# Plot box plots for steps
plt.figure(figsize=(12, 6))
plt.boxplot(
    steps_data,
    labels=grid_labels,
    boxprops=dict(linewidth=2, color='blue'),
    whiskerprops=dict(linewidth=3, color='red'),
    capprops=dict(linewidth=2, color='black'),
    medianprops=dict(linewidth=2, color='green')
)
plt.title("Steps Distribution per Grid Size")
plt.xlabel("Grid Size")
plt.ylabel("Steps")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot box plots for max queue size
plt.figure(figsize=(12, 6))
plt.boxplot(
    queue_data,
    labels=grid_labels,
    boxprops=dict(linewidth=2, color='blue'),
    whiskerprops=dict(linewidth=3, color='red'),
    capprops=dict(linewidth=2, color='black'),
    medianprops=dict(linewidth=2, color='green')
)
plt.title("Max Queue Size Distribution per Grid Size")
plt.xlabel("Grid Size")
plt.ylabel("Max Queue Size")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
