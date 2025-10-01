import random
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

def bfs(start, targets, m, n): #breadth first search function to find shortest path to dirty cell, not fully efficient
    queue = [(start, [])]   # list as queue for future direction
    visited = set([start]) #list as set to track visited cells
    maxq=0
    while queue:
        (x, y), path = queue.pop(0)  # pop first element

        if (x, y) in targets:  # reached a dirty cell
            return path + [(x, y)]

        # Explore neighboring cells
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]: #searches 4 neighboring cells
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited: #adds to queue if valid
                visited.add((nx, ny)) #adds to set of visited cells
                queue.append(((nx, ny), path + [(x, y)])) #adds to queue
                if(len(queue)>maxq):
                    maxq=len(queue)
                    
    return None  

def vacuum_world_bfs(m, n):
    # Ask user for grid size
   
    dirt_prob = 0.5 # Probability of a cell being dirty set as always 50%

    grid = make_random_grid(m, n, dirt_prob) #runs grid function and sets it to variable grid
    agent_pos = (0, 0)

    #print("\nInitial Grid:")
    #print_grid(grid, agent_pos) #initial grid with agent in first cell

    steps = 0 #tracks moves
    cleans = 0 #tracks cleans

    while any(1 in row for row in grid):  #checks for dirty cells
        #Finds the dirty cells
        dirty_cells = {(x, y) for x in range(m) for y in range(n) if grid[x][y] == 1}

        #use bfs to create path to next dirty cell
        path = bfs(agent_pos, dirty_cells, m, n)

        #follows path
        for pos in path:
            agent_pos = pos
            steps += 1 #adds movement if was taken

            #Clean if dirty
            x, y = agent_pos
            if grid[x][y] == 1: #checks if cell is dirty
                grid[x][y] = 0 #cleans cell
                cleans += 1
                action = "Cleaned"
            else:
                action = "Moved"

            #print(f"Step {steps}: Agent moved to {agent_pos} → {action}") #prints action taken
            #print_grid(grid, agent_pos) #reprints grid with new agent location and updated dirt if was cleaned

    #print(f" Vacuum done in {steps} steps with {cleans} cleans") #final efficiency evaluation statement
    return steps/cleans if cleans > 0 else 0

#init worl

initial_m = 10
initial_n = 10

for i in range(10):
    m = initial_m + i * 10
    n = initial_n + i * 10
    result= vacuum_world_bfs(m, n) #runs function
    print(f"Grid Size: {m}x{n}, Steps/Cleans Ratio: {result}") #prints grid size and efficiency ratio)
