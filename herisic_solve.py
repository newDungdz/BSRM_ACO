import time

# Input processing
with open("mini_project/test_case/case4.txt") as case_file:    
    lines = case_file.readlines()
    N, K = map(int, lines[1].split())  # N: customers, K: technicians
    d = list(map(int, lines[2].split()))  # Maintenance times
    t = []
    for line in lines[3:]:
        if(line == "Output:\n"): break
        t.append(list(map(int, line.split())))
    print(N, K)
    print(d)
    print(t)
