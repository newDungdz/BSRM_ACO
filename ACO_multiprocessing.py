import numpy as np
import random, math
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
import os

# Parameters
NUM_ANTS = 10    # Number of ants
MAX_ITER = 10 # Maximum iterations
TAU_0 = 0.01     # Initial pheromone
RHO = 0.1        # Evaporation rate
BETA = 1         # Heuristic weight
Q0_MIN = 0.9     # Exploitation probability min value
Q0 = Q0_MIN      # Exploitation probability 
Q0_MAX = 0.9     # Exploitation probability max value
Q0_STEP = (Q0_MAX - Q0_MIN) / 100  # Exploitation probability step
MAX_TIME = 290   # Maximum time limit for the algorithm
LOCAL_SEARCH = True # Use local search or not
LOCAL_DILEMA = False # If stuck on local min
NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Use all but one CPU core

begin_time = time.time()

def plot_cost_history(history, filename="aco_cost_history.png"):
    """
    Plot the history of best costs and current best costs over iterations.
    
    Args:
        history: Dictionary containing 'best_max_costs', 'current_best_costs', and 'iterations'
        filename: Name of the file to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot both cost histories
    plt.plot(history['iterations'], history['best_max_costs'], 'b-', label='Global Best Cost', linewidth=2)
    plt.plot(history['iterations'], history['current_best_costs'], 'r--', label='Iteration Best Cost', alpha=0.7)
    
    # Add a vertical line for each improvement in the global best cost
    improvements = []
    prev_best = float('inf')
    for i, cost in enumerate(history['best_max_costs']):
        if cost < prev_best:
            improvements.append(i)
            prev_best = cost
    
    for i in improvements:
        plt.axvline(x=history['iterations'][i], color='g', linestyle=':', alpha=0.5)
    
    # Add annotations for significant improvements
    if len(improvements) > 0:
        # Only annotate some points to avoid cluttering (first, last, and some in between)
        to_annotate = [improvements[0], improvements[-1]]
        if len(improvements) > 5:
            to_annotate.append(improvements[len(improvements)//2])
            
        for i in to_annotate:
            iteration = history['iterations'][i]
            cost = history['best_max_costs'][i]
            plt.annotate(f'{cost:.2f}', 
                        xy=(iteration, cost),
                        xytext=(iteration, cost*1.05),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=9)
    
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Max Route Cost')
    plt.title('ACO Algorithm Convergence History')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add text box with summary statistics
    if len(history['best_max_costs']) > 0:
        initial_cost = history['best_max_costs'][0] if history['best_max_costs'][0] != float('inf') else history['current_best_costs'][0]
        final_cost = history['best_max_costs'][-1]
        improvement = ((initial_cost - final_cost) / initial_cost) * 100 if initial_cost > 0 else 0
        
        stats_text = (f"Initial cost: {initial_cost:.2f}\n"
                    f"Final cost: {final_cost:.2f}\n"
                    f"Improvement: {improvement:.2f}%\n"
                    f"Total iterations: {len(history['iterations'])}")
        
        plt.figtext(0.15, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nCost history plot saved as '{filename}'")
    
    # Close the figure to free memory
    plt.close()

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

def ant_construct_solution(n, k, travel_time, service_time, pheromones, eta_matrix, pre_comp_eta_matrix, seed=None):
    """Construct a solution with K routes using ACO (optimized for large N and multiprocessing)."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    unvisited = set(range(1, n + 1))
    solution = []
    current_route = [0]
    return_count = 0
    average_customer = n / k
    customer_offset = 0
    min_max_range = average_customer / 3
    current_cost = 0
    
    # Create a local copy of pheromones for thread safety
    pheromones_local = np.copy(pheromones)
    
    def return_depot():
        nonlocal current_route, return_count, customer_offset, current_cost
        return_count += 1
        customer_offset += len(current_route) - 1 - average_customer
        current_route.append(0)
        solution.append(current_route)
        current_route = [0]
        current_cost = 0
        
    while unvisited:
        current = current_route[-1]
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
        if(len(current_route) == max_customers): 
            eta = pre_comp_eta_matrix[current][next_options_np]
        else: 
            eta = eta_matrix[current][next_options_np]
            
        tau = pheromones_local[current][next_options_np]
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
        pheromones_local[current][next_node] = (1 - RHO) * pheromones_local[current][next_node] + RHO * TAU_0
        
        if next_node == 0:
            return_depot()
        else:
            current_route.append(next_node)
            current_cost = new_cost
            unvisited.discard(next_node)

    if len(current_route) > 1:
        current_route.append(0)
        solution.append(current_route)

    return solution, pheromones_local

def local_search(solution, travel_time, service_time, seed=None):
    """Local search: try swap and relocate to reduce max route cost."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
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
        swap_attempts = min(10, len(route_i) // 2) * (3 if LOCAL_DILEMA else 1)
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

    return best_solution, best_costs, best_max_cost

def process_ant_solution(solution, travel_time, service_time, seed):
    """Process a single ant solution with local search if enabled."""
    if LOCAL_SEARCH:
        solution, costs, max_cost = local_search(solution, travel_time, service_time, seed)
    else:
        costs = [calculate_route_cost(r, travel_time, service_time) for r in solution]
        max_cost = max(costs)
    
    return solution, costs, max_cost

def generate_ant_solution(n, k, travel_time, service_time, pheromones, eta_matrix, pre_comp_eta_matrix, seed):
    """Generate a single ant solution and process it."""
    solution, local_pheromones = ant_construct_solution(n, k, travel_time, service_time, 
                                                       pheromones, eta_matrix, pre_comp_eta_matrix, seed)
    solution, costs, max_cost = process_ant_solution(solution, travel_time, service_time, seed)
    return solution, costs, max_cost, local_pheromones

def acs_vrp(n, k, travel_time, service_time):
    """Main ACO algorithm with multiprocessing."""
    global Q0, LOCAL_DILEMA
    pheromones = np.full((n + 1, n + 1), TAU_0)  # Pheromone matrix
    best_solution = None
    best_max_cost = float('inf')
    eta_matrix = compute_eta_matrix(travel_time, service_time)
    pre_comp_eta_matrix = compute_eta_matrix_precompute(travel_time, service_time)
    unchange_iter = 0
    
    # History tracking for plotting
    history = {
        'iterations': [],
        'best_max_costs': [],
        'current_best_costs': []
    }
    
    # Prepare shared data for multiprocessing
    travel_time_shared = np.array(travel_time)
    service_time_shared = np.array(service_time)
    
    for iteration in range(MAX_ITER):
        if(time.time() - begin_time > MAX_TIME):
            print("\nTime limit exceeded")
            break
            
        start_time = time.time()
        
        # Generate random seeds for reproducibility within processes
        seeds = [np.random.randint(0, 10000) for _ in range(NUM_ANTS)]
        
        # Define the partial function for parallel execution
        generate_func = partial(
            generate_ant_solution, 
            n, k, 
            travel_time_shared, 
            service_time_shared,
            pheromones,
            eta_matrix, 
            pre_comp_eta_matrix
        )
        
        # Execute ant construction and processing in parallel
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            results = list(executor.map(generate_func, seeds))
        
        # Process results
        ant_solutions = []
        ant_costs = []
        ant_max_costs = []
        local_pheromone_updates = []
        
        for solution, costs, max_cost, local_pheromones in results:
            ant_solutions.append(solution)
            ant_costs.append(costs)
            ant_max_costs.append(max_cost)
            local_pheromone_updates.append(local_pheromones)
        
        # Apply local pheromone updates
        for local_pheromone in local_pheromone_updates:
            # Blend local updates into the global pheromone matrix
            # Take the minimum to prevent increasing pheromones
            pheromones = np.minimum(pheromones, local_pheromone)
        
        # Find the best solution in this iteration
        best_ant_idx = np.argmin(ant_max_costs)
        current_best_solution = ant_solutions[best_ant_idx]
        current_best_cost = ant_costs[best_ant_idx]
        current_best_max_cost = ant_max_costs[best_ant_idx]
        
        # Update global best if we found a better solution
        best_buffer = 1
        if current_best_max_cost < best_max_cost * best_buffer:
            best_solution = current_best_solution
            best_max_cost = current_best_max_cost
            unchange_iter = 0
            LOCAL_DILEMA = False
        else:
            unchange_iter += 1
            if(unchange_iter > 20): 
                LOCAL_DILEMA = True
            if(time.time() - begin_time >= MAX_TIME - 10):
                LOCAL_DILEMA = False
        
        # Update history for plotting
        history['iterations'].append(iteration + 1)
        history['best_max_costs'].append(best_max_cost)
        history['current_best_costs'].append(current_best_max_cost)
    
        # Print the current best solution and its cost on the same line
        iteration_info = f"Iteration {iteration + 1}: | Best Max Cost: {best_max_cost:.2f} | Routes: {len(current_best_cost)} | Current: {current_best_max_cost:.2f} | Time: {time.time() - begin_time:.2f}s | Dilemma: {LOCAL_DILEMA}"
        costs_info = f" | Costs: {current_best_cost}"
        route_len_info = f" | Route Lengths: {[len(route) for route in current_best_solution]}"
        print(f"\r{iteration_info}{costs_info}{route_len_info}", end="", flush=True)

        # Global pheromone update
        for route in best_solution:
            for i in range(len(route) - 1):
                pheromones[route[i]][route[i+1]] = (1 - RHO) * pheromones[route[i]][route[i+1]] + RHO / best_max_cost
        
        Q0 = min(Q0 + Q0_STEP, Q0_MAX) 

    # Plot cost history
    plot_cost_history(history)
    
    return best_solution

def main():
    # Set up multiprocessing support for Windows
    if __name__ == '__main__':
        mp.freeze_support()
    
    # Input processing
    try:
        with open("generated_test_cases/small_test.txt") as case_file:    
            lines = case_file.readlines()
            N, K = map(int, lines[1].split())  # N: customers, K: technicians
            d = list(map(int, lines[2].split()))  # Maintenance times
            t = []
            for line in lines[3:]:
                if(line == "Output:\n"): break
                t.append(list(map(int, line.split())))
    except FileNotFoundError:
        print("File not found, trying to read from standard input...")
        N, K = map(int, input().split())  # N: customers, K: technicians
        d = list(map(int, input().split()))  # Maintenance times
        t = [list(map(int, input().split())) for _ in range(N + 1)]  # Travel times
    
    d = [0] + d  # Add depot service time

    print(f"Problem size: {N} customers, {K} technicians")
    print(f"Using {NUM_PROCESSES} processes for parallel execution")
    
    # Create output directory if it doesn't exist
    output_dir = "aco_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run ACO
    solution = acs_vrp(N, K, t, d)

    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save results to text file
    results_file = os.path.join(output_dir, f"aco_results_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write(f"{K}\n")
        for route in solution:
            f.write(f"{len(route) - 1}\n")
            f.write(f"{' '.join(map(str, route))}\n")
        
        # Calculate and write costs
        costs = [calculate_route_cost(r, t, d) for r in solution]
        f.write(f"\nNumber of Routes: {len(costs)}, Max Cost: {max(costs)}\n")
        f.write(f"Route Costs: {costs}\n")
        f.write(f"\nTotal Execution Time: {time.time() - begin_time:.2f} seconds")
    
    print(f"\nResults saved to '{results_file}'")
    
    # Output - add a newline to ensure we start on a fresh line after the progress updates
    print("\n")
    print(K)
    for route in solution:
        print(len(route) - 1)  # Lk: number of customers + depot transitions - 1
        print(" ".join(map(str, route)))

    # Verify costs
    costs = [calculate_route_cost(r, t, d) for r in solution]
    print(f"Number of Route: {len(costs)}, Max Cost: {max(costs)}")
    print(f"Route Costs: {costs}")
    
    # Save cost history plot in the output directory
    plot_filename = os.path.join(output_dir, f"aco_cost_history_{timestamp}.png")
    # Plot is already created in the acs_vrp function, but we can create a new one with the filename
    plt.gcf().savefig(plot_filename)
    print(f"Cost history plot saved as '{plot_filename}'")
    
    # Return the solution and costs for potential further analysis
    return solution, costs

if __name__ == "__main__":
    main()