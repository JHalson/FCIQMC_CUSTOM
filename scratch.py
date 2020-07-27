import numpy as np

nsites = 2


def fill(nsites, states=[], index=0):
    if index > nsites + 1:
        return states
    if index == 0:
        states.append([0]*2*nsites)
    newstates = []
    for state in states:
        newstate = state.copy()
        newstate[index] = 1
        newstates.append(newstate)
    states += newstates
    return fill(nsites, states, index+1)

def genpairs(array):
    out = []
    for item in array:
        for item2 in array:
            out.append([item, item2])
    return out


states = fill(nsites)

print(states)

pairs = genpairs(states)

print(pairs)

H_U = np.zeros(len(states)**2)

for ind, pair in enumerate(pairs):
    #print(pair)
    H_U[ind] = sum( [pair[0][i]*pair[1][i] for i in range(len(pair))])


H_U.reshape(len(states), len(states))
#H += U*H_U

print(H_U)


