from ortools.linear_solver import pywraplp
import time

# Input processing
with open("mini_project/test_case/case5.txt") as case_file:    
    lines = case_file.readlines()
    N, K = map(int, lines[1].split())  # N: customers, K: technicians
    d = list(map(int, lines[2].split()))  # Maintenance times
    t = []
    for line in lines[3:]:
        if(line == "Output:\n"): break
        t.append(list(map(int, line.split())))
    # print(N, K)
    # print(d)
    # print(t)

# Initialize solver
solver = pywraplp.Solver.CreateSolver("CBC")
if not solver:
    exit(0)

# Decision variables
x = {}
for i in range(N+1):
    for j in range(N+1):
        if i != j:
            x[i, j] = solver.IntVar(0, 1, f"x[{i}, {j}]")

y = solver.IntVar(0, solver.infinity(), "y")  # Maximum completion time

# Time variables
mv_time = [solver.IntVar(0, solver.infinity(), f"mv_time[{i}]") for i in range(N+1)]

# Constraints
# Each customer visited exactly once
for j in range(1, N+1):
    solver.Add(sum(x[i, j] for i in range(N+1) if i != j) == 1)  # In-degree
    solver.Add(sum(x[j, i] for i in range(N+1) if i != j) == 1)  # Out-degree

# Number of routes constraint
solver.Add(sum(x[0, j] for j in range(1, N+1)) <= K)
solver.Add(sum(x[0, j] for j in range(1, N+1)) >= 1)

# Time constraints
solver.Add(mv_time[0] == 0)  # Depot start time
M = 10000  # Big-M value

# Time accumulation and subtour elimination
for i in range(N+1):
    for j in range(1, N+1):
        if i != j:
            solver.Add(mv_time[j] >= mv_time[i] + t[i][j] + d[j-1] - M * (1 - x[i, j]))

# Link time to objective
for i in range(1, N+1):
    solver.Add(mv_time[i] + t[i][0] <= y)

# Objective: Minimize maximum completion time
solver.Minimize(y)
start_time = time.time()
print("Number of variables =", solver.NumVariables())
print("Number of constraints =", solver.NumConstraints())
print(f"Solving with {solver.SolverVersion()}")
print("Solving")
# Solve
status = solver.Solve()

# for i in range(N + 1):
#     for j in range(N + 1):
#         if(i != j and x[i, j].solution_value() == 1): print(f"x[{i}, {j}] = " , x[i, j].solution_value())

print(round(y.solution_value()))
# for i in range(N + 1):
#     for j in range(N + 1):
#         if(i != j and x[i, j].solution_value() == 1): print(f"x[{i}, {j}] = " , x[i, j].solution_value())
end_time = time.time()

# Output results
if status == pywraplp.Solver.OPTIMAL:
    print(K)  # Number of technicians
    
    # Extract routes
    routes = []
    visited = set()
    for k in range(K):
        current = 0
        route = [0]
        while True:
            next_node = None
            for j in range(N+1):
                if (current, j) in x and (x[current, j]).solution_value() > 0.5 and j not in visited:
                    next_node = j
                    break
            if next_node is None or next_node == 0:
                route.append(0)
                break
            route.append(next_node)
            visited.add(next_node)
            current = next_node
        if len(route) > 2:  # Only include non-empty routes
            routes.append(route)
    
    # Print routes in required format
    for route in routes:
        print(len(route))  # Lk (number of nodes excluding return to 0)
        print(*route)  # Route points
    
    # Print maximum time (for verification)
    # print(f"Maximum completion time: {round(solver.Objective().Value())}")
else:
    print("No optimal solution found")

print(f"Solve time: {end_time - start_time:.4f} seconds")
