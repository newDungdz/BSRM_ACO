from ACO import acs_vrp
import os

def main():
    # Input processing
    with open("generated_test_cases\geo_test.txt") as case_file:    
        lines = case_file.readlines()
        N, K = map(int, lines[1].split())  # N: customers, K: technicians
        d = list(map(int, lines[2].split()))  # Maintenance times
        t = []
        for line in lines[3:]:
            if(line == "Output:\n"): break
            t.append(list(map(int, line.split())))
    d = [0] + d  # Add depot service time

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