from functools import reduce
import numpy as np

class Hubbard:
    def __init__(self, nsites, t, U, nelec=None, periodic=False):
        self.nsites=nsites
        self.nelec = nelec
        self.periodic = periodic
        self.t = t
        self.U = U
        self.matrix = None
        self.check_sanity()
        self._select_all = reduce(lambda a, b: 2 ** a + b, range(self.nsites))
        self._select_down = self._select_all - reduce(lambda a, b: 2 ** a + b, range(self.nsites // 2))
        self._select_up = self._select_all - self._select_down
        self.index_table = self.make_index_table()
        self.size = len(self.index_table)

    def check_sanity(self):
        if self.nelec:
            assert self.nelec <= self.nsites * 2, "overfull lattice"

    def bstring(self, number, dualchannel=False):
        # string containing bit representation with number of leading zeros approrpiate to system
        fmt = "{{:0{}b}}".format(self.nsites*2 if dualchannel else self.nsites)
        return fmt.format(number)

    def make_index_table(self):
        configs = []
        for config in range(2**(self.nsites*2)):
            if self.nelec and self.bstring(config, True).count('1') != self.nelec:
                continue
            else:
                configs.append(config)
        return {i : configs[i] for i in range(len(configs))}

    def print_index_table(self):
        for ind, config in self.index_table.items():
            print(("Index: {} -> {}  ({})").format(ind, self.bstring(config, True), config))

    def separate_spins(self, state):
        # (down, up)
        return state // 2**(self.nsites), state % 2**(self.nsites)


    def occs_from_ind(self, ind):
        return self.index_table[ind]

    def check_spin_conservation(self, state1, state2):
        sep1 = self.separate_spins(state1)
        sep2 = self.separate_spins(state2)
        cons = True
        for spin in (0,1):
            if sum(map(int, self.bstring(sep1[spin]))) != sum(map(int, self.bstring(sep2[spin]))):
                cons = False
        return cons

    def calc_matrix_element(self, state1, state2):
        # <2|H|1>
        if not self.check_spin_conservation(state1, state2):
            return 0

        sep1 = self.separate_spins(state1)
        sep2 = self.separate_spins(state2)

        if state1 == state2:
            # U-term
            return self.U*bin(sep1[0] & sep1[1]).count('1')
        else:
            # t-term
            H_t = 0
            for spin in zip(sep1,sep2):
                spin1 = self.bstring(spin[0])  # state 1 spin for this channel
                spin2 = self.bstring(spin[1])  # state 2
                for pos in range(self.nsites-1):
                    if spin1[pos] == spin2[pos+1]:
                        H_t += 1
                if self.periodic and (spin1[0] == spin2[-1]):
                    H_t += 1
            if H_t > 1:
                # only one hopping permitted
                return 0
            else:
                return -self.t*H_t

    def build_matrix(self):
        # explicitly construct the Hamiltonian matrix

        if not self.matrix:
            self.matrix = np.zeros((self.size, self.size))
            for i in range(self.size):
                state1 = self.index_table[i]
                for j in range(i+1):
                    state2 = self.index_table[j]
                    self.matrix[i,j] = self.calc_matrix_element(state1, state2)
                    self.matrix[j,i] = self.matrix[i,j]
        return self.matrix

    def get_exact_energy(self):
        self.build_matrix()
        return np.linalg.eigh(self.matrix)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t=1
    nsites=2

    U_data = np.linspace(0.1,10)
    E_data = []
    for i, U in enumerate(U_data):
        H = Hubbard(nsites=nsites,t=t,U=U,nelec=nsites)
        E_data.append(H.get_exact_energy()[0])

    E_data = np.array(E_data)
    for band in range(E_data.shape[1]):
        plt.plot(U_data,E_data[:,band])

    if nsites == 2:
        plt.plot(U_data, U_data, ls="--", label=r"Exact E_2")
        plt.plot(U_data, [0]*len(U_data), ls="--", label=r"Exact E_1")
        plt.plot(U_data, U_data/2 + np.sqrt((U_data/2)**2 + 4*t**2), ls="--", label=r"Exact E_3")
        plt.plot(U_data, U_data / 2 - np.sqrt((U_data / 2) ** 2 + 4 * t ** 2), ls="--", label=r"Exact E_0")
        plt.legend()

    plt.ylabel('Energy')
    plt.xlabel('U/t')
    plt.show()



