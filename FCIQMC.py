import numpy as np
walkers = walkers() # associative array object for walkers

hamiltonian = hamiltonian() # system, stores electron integrals

for timestep in range(maxtime):
    for parent in walkers:
        spawned = walkers()
        for child in range(round(abs(parent.weight))):
            generate_connection(parent)
            # spawning logic
            # capture result in spawned array, keeping det label
        walker_decay()
    merge_walkers()
    collect_stats()







