import numpy as np
import random
import math, os
import matplotlib.pyplot as plt

tc_folder = "generated_test_cases"
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

def plot_points(coords, filename="depot_customer_plot.png"):
    """
    Plot depot and customer points on a 2D scatter plot and save as an image.
    
    Parameters:
    - coords: List of (x, y) tuples for depot + customers. Index 0 is depot.
    - filename: Name of the file to save the plot (default: depot_customer_plot.png)
    """
    x, y = zip(*coords)
    plt.figure(figsize=(6, 6))
    
    # Plot depot (index 0)
    plt.scatter(x[0], y[0], color='red', marker='s', s=50, label='Depot (0)')
    # plt.text(x[0] + 1, y[0], '0', color='red')
    
    # Plot customers (index 1..N)
    for i in range(1, len(coords)):
        plt.scatter(x[i], y[i], color='blue', marker='o', s=20)
        # plt.text(x[i] + 1, y[i], str(i), color='blue')
    
    plt.title("Depot and Customer Locations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    filename = os.path.join(plot_folder, filename)
    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as '{filename}'")
    
    # Close the figure to free memory
    plt.close()

def generate_test_case(N, K, d_range=(1, 100), t_range=(1, 100), seed=None):
    """
    Generate a test case for the network maintenance VRP problem.
    
    Parameters:
    - N: Number of customers
    - K: Number of technicians
    - d_range: Range for service times (min, max)
    - t_range: Range for travel times (min, max)
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary containing the problem data
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate service times for each customer
    d = [random.randint(d_range[0], d_range[1]) for _ in range(N)]
    
    # Generate travel time matrix (t)
    t = np.zeros((N+1, N+1), dtype=int)
    
    # Fill the matrix with random travel times
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                t[i][j] = random.randint(t_range[0], t_range[1])
    
    # Ensure triangle inequality is satisfied
    for k in range(N+1):
        for i in range(N+1):
            for j in range(N+1):
                if t[i][j] > t[i][k] + t[k][j]:
                    t[i][j] = t[i][k] + t[k][j]
    
    return {
        "N": N,
        "K": K,
        "d": d,
        "t": t
    }

def write_test_to_file(test_data, filename):
    """
    Write the test data to a file in the required format.
    
    Parameters:
    - test_data: Dictionary containing the problem data
    - filename: Name of the output file
    """
    N = test_data["N"]
    K = test_data["K"]
    d = test_data["d"]
    t = test_data["t"]
    
    # Create the directory if it doesn't exist
    if not os.path.exists(tc_folder):
        os.makedirs(tc_folder)
    filename = os.path.join(tc_folder, filename)

    with open(filename, 'w') as f:
        f.write("Input:\n")
        # Write N and K
        f.write(f"{N} {K}\n")
        
        # Write service times
        f.write(" ".join(map(str, d)) + "\n")
        
        # Write travel time matrix
        for i in range(N+1):
            f.write(" ".join(map(str, t[i])) + "\n")

def solve_greedy(test_data):
    """
    Solve the problem using a simple greedy algorithm.
    This is not optimal but provides a feasible solution for testing.
    
    Parameters:
    - test_data: Dictionary containing the problem data
    
    Returns:
    - List of routes for each technician
    """
    N = test_data["N"]
    K = test_data["K"]
    d = test_data["d"]
    t = test_data["t"]
    
    # Create a list of customers
    customers = list(range(1, N+1))
    
    # Initialize routes and working times for each technician
    routes = [[] for _ in range(K)]
    working_times = [0] * K
    
    # Sort customers by service time (descending)
    customers.sort(key=lambda x: d[x-1], reverse=True)
    
    # Assign customers to technicians using a greedy approach
    for customer in customers:
        # Find the technician with the least working time
        tech_idx = working_times.index(min(working_times))
        
        # Add the customer to this technician's route
        routes[tech_idx].append(customer)
        
        # Update working time
        if not routes[tech_idx]:  # If this is the first customer
            working_times[tech_idx] = t[0][customer] + d[customer-1] + t[customer][0]
        else:
            prev_customer = routes[tech_idx][-2] if len(routes[tech_idx]) > 1 else 0
            working_times[tech_idx] += t[prev_customer][customer] + d[customer-1]
    
    # Format routes: add depot at the beginning and end
    formatted_routes = []
    for route in routes:
        if route:
            formatted_route = [0] + route + [0]
            formatted_routes.append(formatted_route)
        else:
            # Handle case where a technician might not have any customers
            formatted_routes.append([0, 0])
    
    return formatted_routes

def solve_nearest_neighbor(test_data):
    """
    Solve the problem using the nearest neighbor heuristic.
    Builds routes by always choosing the closest unvisited customer.
    
    Parameters:
    - test_data: Dictionary containing the problem data
    
    Returns:
    - List of routes for each technician
    """
    N = test_data["N"]
    K = test_data["K"]
    d = test_data["d"]
    t = test_data["t"]
    
    # Create a list of unvisited customers
    unvisited = set(range(1, N+1))
    
    # Initialize routes and working times for each technician
    routes = [[] for _ in range(K)]
    working_times = [0] * K
    
    # Assign customers to technicians using nearest neighbor
    current_tech = 0
    
    while unvisited:
        # Find current position of the technician
        current_pos = routes[current_tech][-1] if routes[current_tech] else 0
        
        # Find the nearest unvisited customer
        min_time = float('inf')
        nearest = None
        
        for customer in unvisited:
            travel_time = t[current_pos][customer]
            if travel_time < min_time:
                min_time = travel_time
                nearest = customer
        
        # Add the customer to this technician's route
        routes[current_tech].append(nearest)
        unvisited.remove(nearest)
        
        # Update working time
        if len(routes[current_tech]) == 1:  # If this is the first customer
            working_times[current_tech] = t[0][nearest] + d[nearest-1]
        else:
            prev_customer = routes[current_tech][-2]
            working_times[current_tech] += t[prev_customer][nearest] + d[nearest-1]
        
        # Move to next technician if they have fewer customers
        # This balances the workload
        min_route_length = min(len(route) for route in routes)
        candidate_techs = [i for i, route in enumerate(routes) if len(route) == min_route_length]
        current_tech = min(candidate_techs, key=lambda i: working_times[i])
    
    # Add return to depot and update working times
    for i, route in enumerate(routes):
        if route:
            last_customer = route[-1]
            working_times[i] += t[last_customer][0]  # Add time to return to depot
    
    # Format routes: add depot at the beginning and end
    formatted_routes = []
    for route in routes:
        if route:
            formatted_route = [0] + route + [0]
            formatted_routes.append(formatted_route)
        else:
            # Handle case where a technician might not have any customers
            formatted_routes.append([0, 0])
    
    return formatted_routes

def solve_savings(test_data):
    """
    Solve the problem using Clarke & Wright savings algorithm.
    This algorithm is designed to minimize total distance by maximizing "savings".
    
    Parameters:
    - test_data: Dictionary containing the problem data
    
    Returns:
    - List of routes for each technician
    """
    N = test_data["N"]
    K = test_data["K"]
    d = test_data["d"]
    t = test_data["t"]
    
    # Initialize: each customer is served by a separate technician
    # We'll merge routes until we have K routes
    routes = [[i] for i in range(1, N+1)]
    
    # Calculate savings for each pair of customers
    savings = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            # Savings formula: cost(0,i) + cost(j,0) - cost(i,j)
            saving = t[0][i] + t[j][0] - t[i][j]
            savings.append((saving, i, j))
    
    # Sort savings in descending order
    savings.sort(reverse=True)
    
    # Merge routes based on savings
    while len(routes) > K:
        for saving, i, j in savings:
            # Find routes containing i and j
            route_i = None
            route_j = None
            for idx, route in enumerate(routes):
                if i in route:
                    route_i = idx
                if j in route:
                    route_j = idx
            
            # If i and j are in different routes, merge them
            if route_i is not None and route_j is not None and route_i != route_j:
                # Check if i and j are at the ends of their routes
                i_position = routes[route_i].index(i)
                j_position = routes[route_j].index(j)
                
                can_merge = False
                merged_route = None
                
                # Check different merging scenarios (end to start, start to end, etc.)
                if i_position == len(routes[route_i]) - 1 and j_position == 0:
                    # i is at the end of its route, j is at the start of its route
                    merged_route = routes[route_i] + routes[route_j]
                    can_merge = True
                elif i_position == 0 and j_position == len(routes[route_j]) - 1:
                    # i is at the start of its route, j is at the end of its route
                    merged_route = routes[route_j] + routes[route_i]
                    can_merge = True
                
                if can_merge and merged_route:
                    # Remove the two original routes
                    if route_i > route_j:
                        routes.pop(route_i)
                        routes.pop(route_j)
                    else:
                        routes.pop(route_j)
                        routes.pop(route_i)
                    # Add the merged route
                    routes.append(merged_route)
                    break
    
    # If we ended up with more than K routes, we need to merge some more
    while len(routes) > K:
        # Find the two shortest routes to merge
        route_lengths = [(sum(d[customer-1] for customer in route), i) for i, route in enumerate(routes)]
        route_lengths.sort()
        
        # Merge the two shortest routes
        i, j = route_lengths[0][1], route_lengths[1][1]
        routes[i].extend(routes[j])
        routes.pop(j)
    
    # If we have fewer than K routes, create empty routes
    while len(routes) < K:
        routes.append([])
    
    # Format routes: add depot at the beginning and end
    formatted_routes = []
    for route in routes:
        if route:
            formatted_route = [0] + route + [0]
            formatted_routes.append(formatted_route)
        else:
            # Handle case where a technician has no customers
            formatted_routes.append([0, 0])
    
    return formatted_routes

def solve_min_max_algorithm(test_data):
    """
    Solve the problem focusing specifically on minimizing the maximum working time.
    This algorithm repeatedly tries to balance the workload by moving customers
    from the most loaded technician to less loaded ones.
    
    Parameters:
    - test_data: Dictionary containing the problem data
    
    Returns:
    - List of routes for each technician
    """
    N = test_data["N"]
    K = test_data["K"]
    d = test_data["d"]
    t = test_data["t"]
    
    # Start with a solution from the greedy algorithm
    routes = []
    for r in solve_greedy(test_data):
        # Remove depot from beginning and end
        if len(r) > 2:
            routes.append(r[1:-1])
        else:
            routes.append([])
    
    # Calculate working time for each technician
    def calculate_working_time(route):
        if not route:
            return 0
            
        time = 0
        prev = 0  # Start at depot
        
        for customer in route:
            # Add travel time to customer
            time += t[prev][customer]
            # Add service time at customer
            time += d[customer-1]
            # Update previous location
            prev = customer
        
        # Add travel time back to depot
        time += t[prev][0]
        
        return time
    
    working_times = [calculate_working_time(route) for route in routes]
    
    # Iteratively improve the solution
    improved = True
    iterations = 0
    max_iterations = 100  # Prevent infinite loops
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Find the most loaded technician
        max_time_idx = working_times.index(max(working_times))
        
        # Try to move each customer to a different technician
        if not routes[max_time_idx]:
            continue
            
        for i, customer in enumerate(routes[max_time_idx]):
            best_improvement = 0
            best_move = None
            
            # Calculate current working time with this customer
            original_time = working_times[max_time_idx]
            
            # Calculate new working time if we remove this customer
            new_route = routes[max_time_idx][:i] + routes[max_time_idx][i+1:]
            new_time = calculate_working_time(new_route)
            
            for j in range(K):
                if j == max_time_idx:
                    continue
                
                # Try inserting the customer at each possible position in the other route
                for pos in range(len(routes[j]) + 1):
                    test_route = routes[j][:pos] + [customer] + routes[j][pos:]
                    test_time = calculate_working_time(test_route)
                    
                    # Check if this move improves the maximum working time
                    new_max = max(new_time, test_time)
                    improvement = original_time - new_max
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (j, pos)
            
            # If we found a beneficial move, make it
            if best_improvement > 0 and best_move:
                target_tech, pos = best_move
                
                # Remove customer from current route
                customer = routes[max_time_idx].pop(i)
                
                # Add customer to target route
                routes[target_tech].insert(pos, customer)
                
                # Update working times
                working_times[max_time_idx] = calculate_working_time(routes[max_time_idx])
                working_times[target_tech] = calculate_working_time(routes[target_tech])
                
                improved = True
                break
    
    # Format routes: add depot at the beginning and end
    formatted_routes = []
    for route in routes:
        if route:
            formatted_route = [0] + route + [0]
            formatted_routes.append(formatted_route)
        else:
            # Handle case where a technician has no customers
            formatted_routes.append([0, 0])
    
    return formatted_routes

def calculate_max_working_time(routes, test_data):
    """
    Calculate the maximum working time among all technicians.
    
    Parameters:
    - routes: List of routes for each technician
    - test_data: Dictionary containing the problem data
    
    Returns:
    - Maximum working time
    """
    d = test_data["d"]
    t = test_data["t"]
    
    max_time = 0
    
    for route in routes:
        time = 0
        for i in range(len(route) - 1):
            current = route[i]
            next_point = route[i+1]
            
            # Add travel time
            time += t[current][next_point]
            
            # Add service time (if not at depot)
            if next_point != 0:
                time += d[next_point-1]
        
        max_time = max(max_time, time)
    
    return max_time

def write_solution_to_file(routes, filename):
    """
    Write the solution to a file in the required format.
    
    Parameters:
    - routes: List of routes for each technician
    - filename: Name of the output file
    """
    K = len(routes)
    
    with open(filename, 'w') as f:
        # Write K
        f.write(f"{K}\n")
        
        # Write each route
        for k in range(K):
            route = routes[k]
            Lk = len(route) - 1  # -1 because r[0] = r[Lk] = 0
            
            f.write(f"{Lk}\n")
            f.write(" ".join(map(str, route)) + "\n")

def generate_geom_test_case(N, K, grid_size=100, d_range=(1, 100), speed=1.0, seed=None):
    """
    Generate a test case with geometric distances.
    
    Parameters:
    - N: Number of customers
    - K: Number of technicians
    - grid_size: Size of the grid for positioning
    - d_range: Range for service times (min, max)
    - speed: Travel speed factor
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary containing the problem data
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate positions for depot (0) and customers (1 to N)
    positions = []
    
    # Depot at the center
    depot_pos = (grid_size/2, grid_size/2)
    positions.append(depot_pos)
    
    # Customer positions (random)
    for _ in range(N):
        x = random.uniform(0, grid_size)
        y = random.uniform(0, grid_size)
        positions.append((x, y))
    
    # Generate service times for each customer
    d = [random.randint(d_range[0], d_range[1]) for _ in range(N)]
    
    # Generate travel time matrix based on Euclidean distances
    t = np.zeros((N+1, N+1), dtype=int)
    
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                # Calculate Euclidean distance
                dist = math.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                # Convert to travel time
                t[i][j] = max(1, int(dist / speed))
    plot_points(positions)

    return {
        "N": N,
        "K": K,
        "d": d,
        "t": t,
        "positions": positions
    }

def generate_clustered_test_case(N, K, num_clusters=3, grid_size=100, d_range=(1, 100), speed=1.0, seed=None):
    """
    Generate a test case with customers clustered in areas.
    
    Parameters:
    - N: Number of customers
    - K: Number of technicians
    - num_clusters: Number of customer clusters
    - grid_size: Size of the grid for positioning
    - d_range: Range for service times (min, max)
    - speed: Travel speed factor
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary containing the problem data
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate positions for depot (0) and customers (1 to N)
    positions = []
    
    # Depot at the center
    depot_pos = (grid_size/2, grid_size/2)
    positions.append(depot_pos)
    
    # Generate cluster centers
    cluster_centers = []
    for _ in range(num_clusters):
        x = random.uniform(0, grid_size)
        y = random.uniform(0, grid_size)
        cluster_centers.append((x, y))
    
    # Assign customers to clusters
    customers_per_cluster = N // num_clusters
    remaining = N % num_clusters
    
    for i in range(num_clusters):
        cluster_size = customers_per_cluster + (1 if i < remaining else 0)
        center_x, center_y = cluster_centers[i]
        
        for _ in range(cluster_size):
            # Generate position around cluster center
            radius = random.uniform(0, grid_size/10)
            angle = random.uniform(0, 2 * math.pi)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Ensure position is within grid bounds
            x = max(0, min(grid_size, x))
            y = max(0, min(grid_size, y))
            
            positions.append((x, y))
    
    # Generate service times for each customer
    d = [random.randint(d_range[0], d_range[1]) for _ in range(N)]
    
    # Generate travel time matrix based on Euclidean distances
    t = np.zeros((N+1, N+1), dtype=int)
    
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                # Calculate Euclidean distance
                dist = math.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                # Convert to travel time
                t[i][j] = max(1, int(dist / speed))
    
    plot_points(positions)
    return {
        "N": N,
        "K": K,
        "d": d,
        "t": t,
        "positions": positions,
        "cluster_centers": cluster_centers
    }

def generate_mixed_test_case(
    N, K, cluster_ratio=0.5, num_clusters=3, grid_size=100, 
    d_range=(1, 100), speed=1.0, depot_strategy="center", seed=None
):
    """
    Generate a VRP test case with a mix of clustered and random customers.
    
    Parameters:
    - N: Total number of customers
    - K: Number of technicians
    - cluster_ratio: Fraction of customers to place in clusters
    - num_clusters: Number of customer clusters
    - grid_size: Size of the grid
    - d_range: Range for customer service times
    - speed: Speed factor to convert distance to travel time
    - depot_strategy: 'center', 'random', 'outskirt', or 'crowded'
    - seed: Random seed
    
    Returns:
    - Dictionary containing VRP data
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clustered_N = int(N * cluster_ratio)
    random_N = N - clustered_N

    cluster_centers = []
    clustered_positions = []
    random_positions = []

    # Create cluster centers
    for _ in range(num_clusters):
        cx = random.uniform(0, grid_size)
        cy = random.uniform(0, grid_size)
        cluster_centers.append((cx, cy))

    # Assign customers to clusters
    customers_per_cluster = clustered_N // num_clusters
    remaining = clustered_N % num_clusters
    cluster_sizes = [customers_per_cluster + (1 if i < remaining else 0) for i in range(num_clusters)]

    for i, size in enumerate(cluster_sizes):
        cx, cy = cluster_centers[i]
        for _ in range(size):
            radius = random.uniform(0, grid_size / 10)
            angle = random.uniform(0, 2 * math.pi)
            x = max(0, min(grid_size, cx + radius * math.cos(angle)))
            y = max(0, min(grid_size, cy + radius * math.sin(angle)))
            clustered_positions.append((x, y))

    # Add random customers
    for _ in range(random_N):
        x = random.uniform(0, grid_size)
        y = random.uniform(0, grid_size)
        random_positions.append((x, y))

    # Combine all customer positions
    customer_positions = clustered_positions + random_positions

    # Determine depot position
    if depot_strategy == "center":
        depot_pos = (grid_size / 2, grid_size / 2)

    elif depot_strategy == "random":
        depot_pos = (random.uniform(0, grid_size), random.uniform(0, grid_size))

    elif depot_strategy == "crowded" and cluster_centers:
        # Choose the most crowded cluster center
        max_cluster_idx = np.argmax(cluster_sizes)
        depot_pos = cluster_centers[max_cluster_idx]

    elif depot_strategy == "outskirt":
        # Divide the map into a 5x5 grid
        cell_size = grid_size / 5
        grid_counts = [[0 for _ in range(5)] for _ in range(5)]

        # Count how many customers are in each cell
        for x, y in customer_positions:
            col = min(4, int(x / cell_size))
            row = min(4, int(y / cell_size))
            grid_counts[row][col] += 1

        # Find cells with lowest density but non-zero
        min_count = min(c for row in grid_counts for c in row if c > 0)
        sparse_cells = [(r, c) for r in range(5) for c in range(5) if grid_counts[r][c] == min_count]

        # Choose one of the sparse cells randomly
        r, c = random.choice(sparse_cells)
        depot_x = random.uniform(c * cell_size, (c + 1) * cell_size)
        depot_y = random.uniform(r * cell_size, (r + 1) * cell_size)
        depot_pos = (depot_x, depot_y)

    else:
        raise ValueError(f"Unsupported depot_strategy: {depot_strategy}")

    # Combine depot and customers
    positions = [depot_pos] + customer_positions

    # Generate service times (for customers only)
    d = [random.randint(d_range[0], d_range[1]) for _ in range(N)]

    # Generate travel time matrix
    t = np.zeros((N + 1, N + 1), dtype=int)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                t[i][j] = max(1, int(dist / speed))
    return {
        "N": N,
        "K": K,
        "d": d,
        "t": t,
        "positions": positions,
        "cluster_centers": cluster_centers,
        "depot_strategy": depot_strategy
    }

def write_test_case_and_solution_to_file(test_data, solution, filename):
    """
    Write the test case and solution to a file in the required format.
    
    Parameters:
    - test_data: Dictionary containing the problem data
    - solution: List of routes for each technician
    - filename: Name of the output file
    """
    N = test_data["N"]
    K = test_data["K"]
    d = test_data["d"]
    t = test_data["t"]
    
    # Create the directory if it doesn't exist
    if not os.path.exists(tc_folder):
        os.makedirs(tc_folder)
    filename = os.path.join(tc_folder, filename)
    
    num_routes = len(solution)

    with open(filename, 'w') as f:
        f.write("Input:\n")
        # Write N and K
        f.write(f"{N} {K}\n")
        
        # Write service times
        f.write(" ".join(map(str, d)) + "\n")
        
        # Write travel time matrix
        for i in range(N+1):
            f.write(" ".join(map(str, t[i])) + "\n")
        # Write K
        f.write("Output:\n")
        f.write(f"{num_routes}\n")
        
        # Write each route
        for k in range(num_routes):
            route = solution[k]
            Lk = len(route) - 1  # -1 because r[0] = r[Lk] = 0
            
            f.write(f"{Lk}\n")
            f.write(" ".join(map(str, route)) + "\n")

def generate(N , K):
    def calculate_controlled_num_clusters(N, cluster_ratio, 
                                        min_avg_cust_per_cluster=3,
                                        sqrt_factor_min=0.6, 
                                        sqrt_factor_max=1.4):
        """
        Calculates the number of clusters in a controlled way.

        It aims for a number of clusters proportional to the square root of the 
        number of clustered customers, while also ensuring clusters are not too sparse
        (based on min_avg_cust_per_cluster) and handling edge cases.

        Parameters:
        - N: Total number of customers.
        - cluster_ratio: Fraction of customers to place in clusters.
        - min_avg_cust_per_cluster: Desired minimum average customers per cluster.
                                    This helps prevent clusters from being too sparse. (e.g., 3-5)
        - sqrt_factor_min: Multiplier for the lower bound based on sqrt(clustered_N) (e.g., 0.6).
        - sqrt_factor_max: Multiplier for the upper bound based on sqrt(clustered_N) (e.g., 1.4).

        Returns:
        - int: The number of clusters.
        """
        if N == 0:
            return 0  # No customers, so no clusters

        clustered_N = int(N * cluster_ratio)

        if clustered_N == 0:
            return 0  # No customers designated for clustering

        # If fewer customers are to be clustered than the minimum desired per cluster,
        # and there are customers to cluster, put them all in one cluster.
        if 0 < clustered_N < min_avg_cust_per_cluster:
            return 1

        # Base number of clusters around sqrt of the number of clustered customers
        sqrt_clustered_N = math.sqrt(clustered_N)
        
        min_potential_clusters = math.floor(sqrt_clustered_N * sqrt_factor_min)
        max_potential_clusters = math.ceil(sqrt_clustered_N * sqrt_factor_max)

        # Cap the maximum number of clusters to ensure clusters aren't too sparse
        # (each cluster having at least min_avg_cust_per_cluster on average).
        # If min_avg_cust_per_cluster is 0 or negative, this cap is effectively disabled.
        if min_avg_cust_per_cluster > 0:
            cap_max_by_density = clustered_N // min_avg_cust_per_cluster
            # Ensure cap is at least 1 if there are customers to cluster
            cap_max_by_density = max(1, cap_max_by_density) 
        else: 
            # No density constraint, so at most, one customer per cluster
            cap_max_by_density = clustered_N 

        # Determine the actual maximum number of clusters
        # It's the minimum of the sqrt-based upper bound, the density cap, and total clustered customers.
        actual_max_clusters = min(max_potential_clusters, cap_max_by_density)
        actual_max_clusters = min(actual_max_clusters, clustered_N) # Cannot have more clusters than points
        actual_max_clusters = max(1, actual_max_clusters) # Must be at least 1 cluster if clustered_N > 0

        # Determine the actual minimum number of clusters
        actual_min_clusters = max(1, min_potential_clusters)
        # Ensure min_clusters is not greater than max_clusters (can happen if caps are restrictive)
        actual_min_clusters = min(actual_min_clusters, actual_max_clusters) 

        return random.randint(actual_min_clusters, actual_max_clusters)
    seed = random.randint(1, 1000)
    cluster_ratio = random.uniform(0.1, 0.9)
    num_clusters = calculate_controlled_num_clusters(N, cluster_ratio)
    depot_strategy = random.choice(['outskirt', 'crowded'])
    print(f"Generating test case with N={N}, K={K}, cluster_ratio={cluster_ratio}, num_clusters={num_clusters}, depot_strategy={depot_strategy}, seed={seed}")
    test_case = generate_mixed_test_case(
        N=N, K=K, cluster_ratio=cluster_ratio, num_clusters=num_clusters,
        grid_size=100, d_range=(1, 100), speed=1.0, depot_strategy=depot_strategy, seed=seed
    )
    solution = solve_min_max_algorithm(test_case)
    write_test_case_and_solution_to_file(test_case, solution, f"solution_{N}_{K}_{depot_strategy}.txt")
    plot_points(test_case["positions"], filename=f"test_case_{N}_{K}_{depot_strategy}.png")

# Example usage
if __name__ == "__main__":
    # Small test case
    # depot_strategy: 'center', 'random', 'outskirt', or 'crowded'
    N_list = [5, 10, 20, 50, 100, 200, 500, 700, 900, 1000]
    K_div_list = [10, 20]
    for N in N_list:
        if N <= 10:
            K = N // 2
            generate(N, K)
        elif N <= 100:
            K = N // 10
            generate(N, K)
        else:
            for K_div in K_div_list:
                K = N // K_div
                generate(N, K)
    # test_case = generate_mixed_test_case(N=200, K=10, cluster_ratio=0.6, num_clusters=5, depot_strategy='random', seed=123)
    # write_test_case_and_solution_to_file(test_case, solve_min_max_algorithm(test_case), "clustered_test.txt")