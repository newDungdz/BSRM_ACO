import numpy as np
import random, math
import time

# Parameters
NUM_ANTS = 10    # Number of ants
MAX_ITER = 500 # Maximum iterations
TAU_0 = 0.01     # Initial pheromone
PEN = 0.01      # Offset route penalty
RHO = 0.1        # Evaporation rate
BETA = 1         # Heuristic weight
Q0_MIN = 0.1         # Exploitation probability min value
Q0 = Q0_MIN     # Exploitation probability 
Q0_MAX = 0.9         # Exploitation probability max value
Q0_STEP = (Q0_MAX - Q0_MIN) / 60  # Exploitation probability step
MAX_TIME = 290  # Maximum time limit for the algorithm
LOCAL_SEARCH = True # Use local search or not
LOCAL_DILEMA = False # If stuck on local min

best_average_cost = 8000
begin_time = time.time()

def calculate_route_cost(route, travel_time, service_time):
    """Calculate total duration (travel + service) of a route."""
    cost = 0
    for i in range(len(route) - 1):
        cost += travel_time[route[i]][route[i+1]]
        if route[i] != 0:  # Add service time for non-depot nodes
            cost += service_time[route[i]]
    return cost

def compute_eta_matrix(travel_time, service_time):
    n = len(travel_time) - 1
    eta = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            denom = travel_time[i][j] + (service_time[j] if j != 0 else 0)
            eta[i][j] = 0 if denom == 0 else 1 / denom
    return eta

def compute_eta_matrix_precompute(travel_time, service_time):
    """Specialized eta matrix for precomputation: 1 / (travel + service + return to depot)."""
    n = len(travel_time) - 1
    eta = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            denom = travel_time[i][j] + (service_time[j] + travel_time[j][0] if j != 0 else 0)
            eta[i][j] = 0 if denom == 0 else 1 / denom
    return eta

# def calculate_route_cost(route, travel_time, service_time):
#     """Calculate total duration (travel + service) of a route using vectorization."""
#     # Convert route to NumPy array for efficient indexing
#     route = np.array(route)
    
#     # Travel cost: sum of travel_time[i][j] for consecutive pairs
#     travel_cost = np.sum(travel_time[route[:-1], route[1:]])
    
#     # Service cost: sum of service times for non-depot nodes
#     service_cost = np.sum(service_time[route[1:-1]])  # Exclude first and last (depot)
    
#     return travel_cost + service_cost

def ant_construct_solution(n, k, travel_time, service_time, pheromones, eta_matrix, pre_comp_eta_matrix):
    """Construct a solution with K routes using ACO (optimized for large N)."""
    unvisited = set(range(1, n + 1))
    solution = []
    current_route = [0]
    return_count = 0
    average_customer = n / k
    customer_offset = 0
    min_max_range = average_customer / 3
    current_cost = 0
    def return_depot():
        nonlocal current_route, return_count, customer_offset, current_cost
        return_count += 1
        customer_offset += len(current_route) - 1 - average_customer
        # print(current_route , "Len: ", len(current_route) - 1, "Return Count: ", return_count)
        current_route.append(0)
        solution.append(current_route)
        current_route = [0]
        current_cost = 0
        # print("Offset: ", customer_offset, "Min: ", min_customers, "Max: ", max_customers)
    while unvisited:
        current = current_route[-1]
        # Dynamic range update based on remaining workload
        # print("Target: ", dynamic_target, "Min: ", min_customers, "Max: ", max_customers)
        #  * (Q0 - Q0_MIN)
        min_customers = average_customer - customer_offset - round(min_max_range)
        max_customers = average_customer - customer_offset + round(min_max_range)
        if (len(current_route) - 1 >= max_customers and return_count < k - 1):
            return_depot()
            continue

        if return_count < k - 1 and len(current_route) - 1 > min_customers:
            next_options = list(unvisited) + [0]
        else:
            next_options = list(unvisited)
        
        if not next_options:
            current_route.append(0)
            solution.append(current_route)
            break

        next_options_np = np.array(next_options)
        if(len(current_route) == max_customers): eta = pre_comp_eta_matrix[current][next_options_np]
        else: eta = eta_matrix[current][next_options_np]
        tau = pheromones[current][next_options_np]
        tau_eta = tau * (eta ** BETA)

        # Depot boosting
        if 0 in next_options:
            BOOST = 0.5
            depot_idx = np.nonzero(next_options_np == 0)[0][0]
            tau_eta[depot_idx] *= (1 + BOOST * (len(current_route) - 1))

        total = np.sum(tau_eta)
        rng = np.random.default_rng()  # No seed for true randomness
        if total == 0:
            next_node = int(rng.choice(next_options_np))
        elif random.random() < Q0:
            next_node = int(next_options_np[np.argmax(tau_eta)])
        else:
            probs = tau_eta / total
            next_node = int(rng.choice(next_options_np, p=probs))

        # ----- Compute current cost before update -----
        travel = travel_time[current][next_node]
        service = service_time[next_node] if next_node != 0 else 0
        new_cost = current_cost + travel + service
        
        # Local pheromone update
        # pheromones[current][next_node] = (1 - RHO) * pheromones[current][next_node] + RHO * TAU_0
        # if next_node == 0:
        #     # Favor depot returns near target length
        #     pheromone_delta = PEN * (1 + max(0, (average_customer - (len(current_route) - 1)) / average_customer))
        #     pheromones[current][next_node] = (1 - RHO) * pheromones[current][next_node] + RHO * pheromone_delta
        # else:
        #     # Penalize customer moves beyond target
        #     pheromone_delta = PEN * (1 - max(0, (len(current_route) - 1 - average_customer)) / average_customer)
        #     pheromones[current][next_node] = (1 - RHO) * pheromones[current][next_node] + RHO * pheromone_delta
        if next_node == 0:
            # Favor depot returns near the best average cost
            cost_diff = max(0, best_average_cost - new_cost)
            pheromone_delta = PEN * (1 + cost_diff / best_average_cost)
            pheromones[current][next_node] = (1 - RHO) * pheromones[current][next_node] + RHO * pheromone_delta
        else:
            # Penalize moves that cause route cost to exceed best_average
            cost_excess = max(0, new_cost - best_average_cost)
            pheromone_delta = PEN * (1 - cost_excess / best_average_cost)
            pheromones[current][next_node] = (1 - RHO) * pheromones[current][next_node] + RHO * pheromone_delta


        if next_node == 0:
            return_depot()
        else:
            current_route.append(next_node)
            current_cost = new_cost
            unvisited.discard(next_node)

    if len(current_route) > 1:
        current_route.append(0)
        solution.append(current_route)

    return solution

def local_search(solution, travel_time, service_time):
    """Local search: try swap and relocate to reduce max route cost."""
    best_solution = [r.copy() for r in solution]
    best_costs = [calculate_route_cost(r, travel_time, service_time) for r in best_solution]
    best_max_cost = max(best_costs)

    n_routes = len(solution)

    # Pre-calculate route costs for dynamic programming
    route_cost_cache = {i: best_costs[i] for i in range(n_routes)}
        
    # Sort routes by cost (from low to high)
    costs = [route_cost_cache[i] for i in range(n_routes)]
    sorted_indices = sorted(range(n_routes), key=lambda i: costs[i])

    left = 0
    right = n_routes - 1
    swapped_pairs = set()
    
    if LOCAL_DILEMA:
        # Try all possible swaps between every route pair
        for i in range(n_routes):
            for j in range(i + 1, n_routes):
                route_i = solution[i][1:-1]  # Exclude depots
                route_j = solution[j][1:-1]  # Exclude depots

                if not route_i or not route_j:
                    continue

                for ni in route_i:
                    for nj in route_j:
                        pair = tuple(sorted((ni, nj)))
                        if pair in swapped_pairs:
                            continue
                        swapped_pairs.add(pair)

                        new_route_i = [0] + [n if n != ni else nj for n in route_i] + [0]
                        new_route_j = [0] + [n if n != nj else ni for n in route_j] + [0]
                        new_solution = [r.copy() for r in solution]
                        new_solution[i] = new_route_i
                        new_solution[j] = new_route_j

                        cost_i = calculate_route_cost(new_route_i, travel_time, service_time)
                        cost_j = calculate_route_cost(new_route_j, travel_time, service_time)

                        new_costs = []
                        for k in range(n_routes):
                            if k == i:
                                new_costs.append(cost_i)
                            elif k == j:
                                new_costs.append(cost_j)
                            else:
                                new_costs.append(route_cost_cache[k])

                        max_cost = max(new_costs)
                        if max_cost < best_max_cost:
                            best_solution = new_solution
                            best_costs = new_costs
                            best_max_cost = max_cost
                            route_cost_cache[i] = cost_i
                            route_cost_cache[j] = cost_j
    else:
        while left < right:
            i = sorted_indices[right]  # higher cost route
            j = sorted_indices[left]   # lower cost route

            route_i = solution[i][1:-1]
            route_j = solution[j][1:-1]
            if not route_i or not route_j:
                left += 1
                right -= 1
                continue
            # Limit swap attempts per pair
            swap_attempts = min(10, len(route_i) // 2)
            for _ in range(swap_attempts):
                ni = max(route_i, key=lambda c: service_time[c])
                nj = random.choice(route_j)
                pair = tuple(sorted((ni, nj)))
                if pair in swapped_pairs:
                    continue
                swapped_pairs.add(pair)

                new_route_i = [0] + [n if n != ni else nj for n in route_i] + [0]
                new_route_j = [0] + [n if n != nj else ni for n in route_j] + [0]
                new_solution = solution.copy()
                new_solution[i] = new_route_i
                new_solution[j] = new_route_j

                cost_i = calculate_route_cost(new_route_i, travel_time, service_time)
                cost_j = calculate_route_cost(new_route_j, travel_time, service_time)

                new_costs = []
                for k in range(n_routes):
                    if k == i:
                        new_costs.append(cost_i)
                    elif k == j:
                        new_costs.append(cost_j)
                    else:
                        new_costs.append(route_cost_cache[k])

                max_cost = max(new_costs)
                if max_cost < best_max_cost:
                    best_solution = new_solution
                    best_costs = new_costs
                    best_max_cost = max_cost
                    route_cost_cache[i] = cost_i
                    route_cost_cache[j] = cost_j

            left += 1
            right -= 1


    # --- Relocate move from most time consuming route to others ---
    costs = [route_cost_cache[i] for i in range(n_routes)]
    longest_index = np.argmax(costs)
    longest_route = best_solution[longest_index][1:-1]  # Exclude depots

    for customer in longest_route:
        for j in range(n_routes):
            if j == longest_index:
                continue

            for pos in range(1, len(best_solution[j])):  # Insert before depot
                new_solution = [r.copy() for r in best_solution]

                # Remove customer from longest
                new_solution[longest_index] = [n for n in new_solution[longest_index] if n != customer]
                # Insert into new route
                new_solution[j] = new_solution[j][:-1] + [customer] + [0]

                # If source route is empty (just depot), skip (or optionally delete it)
                if len(new_solution[longest_index]) <= 2:
                    continue

                # Only recalculate changed routes
                cost_long = calculate_route_cost(new_solution[longest_index], travel_time, service_time)
                cost_j = calculate_route_cost(new_solution[j], travel_time, service_time)

                new_costs = []
                for k in range(n_routes):
                    if k == longest_index:
                        new_costs.append(cost_long)
                    elif k == j:
                        new_costs.append(cost_j)
                    else:
                        new_costs.append(route_cost_cache[k])

                max_cost = max(new_costs)
                if max_cost < best_max_cost:
                    best_solution = new_solution
                    best_costs = new_costs
                    best_max_cost = max_cost
                    route_cost_cache[longest_index] = cost_long
                    route_cost_cache[j] = cost_j
                    break  # Accept-first strategy


    return best_solution


def acs_vrp(n, k, travel_time, service_time):
    """Main ACO algorithm."""
    global Q0, LOCAL_DILEMA, cur_average_cost
    pheromones = np.full((n + 1, n + 1), TAU_0)  # Pheromone matrix
    best_solution = None
    best_max_cost = float('inf')
    eta_matrix = compute_eta_matrix(travel_time, service_time)
    pre_comp_eta_matrix = compute_eta_matrix_precompute(travel_time, service_time)
    unchange_iter = 0
    for iteration in range(MAX_ITER):
        if(time.time() - begin_time > MAX_TIME):
            # print("Time limit exceeded")
            break
        start_time = time.time()
        ant_solutions = [ant_construct_solution(n, k, travel_time, service_time, pheromones, eta_matrix, pre_comp_eta_matrix) 
                        for _ in range(NUM_ANTS)]
        # print("Ant Contruct Time:", f"{time.time() - start_time:.2f}s")
        # print(f"Ant Solutions: {ant_solutions}")
        current_best_max_cost = float('inf')
        
        # Evaluate and improve solutions
        for solution in ant_solutions:
            if LOCAL_SEARCH:
                solution = local_search(solution, travel_time, service_time)
                # print("Local search Time:", f"{time.time() - start_time:.2f}s")
            costs = [calculate_route_cost(r, travel_time, service_time) for r in solution]
            max_cost = max(costs)
            if max_cost < current_best_max_cost:
                current_best_max_cost = max_cost
                current_best_cost = costs
                current_best_solution = solution
                current_average = sum(costs) / k
            best_buffer = 1
        if current_best_max_cost < best_max_cost * best_buffer:
            best_solution = current_best_solution
            best_max_cost = current_best_max_cost
            best_average_cost = current_average
            unchange_iter = 0
            LOCAL_DILEMA = False
        else:
            unchange_iter += 1
            if(unchange_iter > 30): 
                LOCAL_DILEMA = True
            if(time.time() - begin_time < MAX_TIME - 10):
                LOCAL_DILEMA = False
    
        
        # Print the current best solution and its cost
        # print(f"Iteration {iteration + 1}: | Best Max Cost: {best_max_cost} | Current Best: {current_best_max_cost} | Average: {current_average} | Total Time: {time.time() - begin_time:.2f}s | Local Dilema: {LOCAL_DILEMA}")
        # print(f"All Cost: {current_best_cost}")
        # print(f"Route Ln: {[len(cost) for cost in current_best_solution]}")

        # Global pheromone update
        for route in best_solution:
            for i in range(len(route) - 1):
                pheromones[route[i]][route[i+1]] = (1 - RHO) * pheromones[route[i]][route[i+1]] + RHO / best_max_cost
        
        Q0 = min(Q0 + Q0_STEP, Q0_MAX) 
        
        # Print the pheromone matrix
        # print("  Pheromone Matrix:")
        # print(pheromones)

    return best_solution

def main():
    # Input processing

    # N, K = map(int, input().split())  # N: customers, K: technicians
    # d = list(map(int, input().split()))  # Maintenance times
    # t = [list(map(int, input().split())) for _ in range(N + 1)]  # Travel times
    with open("mini_project/test_case/case3.txt") as case_file:    
        lines = case_file.readlines()
        N, K = map(int, lines[1].split())  # N: customers, K: technicians
        d = list(map(int, lines[2].split()))  # Maintenance times
        t = []
        for line in lines[3:]:
            if(line == "Output:\n"): break
            t.append(list(map(int, line.split())))
    d = [0] + d  # Add depot service time
    # Convert to NumPy arrays
    # d = np.array([0] + d)  # Add depot service time and convert to NumPy array
    # t = np.array(t)        # Convert travel time matrix to NumPy array        
    # print(N, K)
    # print(d)
    # print(t)

    # Run ACO
    solution = acs_vrp(N, K, t, d)

    # Output
    print(K)
    for route in solution:
        print(len(route))  # Lk: number of customers + depot transitions - 1
        print(" ".join(map(str, route)))

    # # Verify costs
    # costs = [calculate_route_cost(r, t, d) for r in solution]
    # print(f"Number of Route: {len(costs)}, Max Cost: {max(costs)}")
    # print(f"\nRoute Costs: {costs}")

if __name__ == "__main__":
    main()