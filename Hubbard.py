from functools import reduce
import numpy as np

class Hubbard:
    def __init__(self, nsites, t, U, nelec=None, spin=None, periodic=False):
        self.nsites=nsites
        self.nelec = nelec
        self.periodic = periodic
        self.spin = spin
        self.t = t
        self.U = U
        self._matrix = None
        self.make_special_integers()
        self.index_table = self.make_index_table()
        self.size = len(self.index_table)
        self.matrix = lambda: self.build_matrix()
        self.check_sanity()

    def make_special_integers(self):
        # function to make integers for bit operations during matrix element evaluation

        # creates a number that on bitwise "and" will deselect the middle two bits
        # to ensure the 'swapped pair' is not the middle two.
        self.exclude_middle = int("1" * (self.nsites - 1) + "00" + "1" * (self.nsites - 1), 2)

        self.select_beta = int("1"*self.nsites,2)
        self.select_alpha = int("1"* self.nsites + "0"*self.nsites, 2)

        # number to allow for periodic boundary conditions
        self.periodic_select = int(("1" + "0"*(self.nsites - 2) + "1"), 2)



    def check_sanity(self):
        if self.nelec:
            assert self.nelec <= self.nsites * 2, "overfull lattice"
        if self._matrix is not None:
            assert np.allclose(np.conj(self._matrix.T), self._matrix), "non-hermitian matrix"

    def bstring(self, number, dualchannel=False):
        # string containing bit representation with number of leading zeros approrpiate to system
        fmt = "{{:0{}b}}".format(self.nsites*2 if dualchannel else self.nsites)
        return fmt.format(number)

    def make_index_table(self):
        configs = []
        for config in range(2**(self.nsites*2)):
            if self.nelec and self.bstring(config, True).count('1') != self.nelec:
                # electron number is conserved
                continue
            if self.spin is not None:
                # check spin conserved
                spins = self.separate_spins(config)
                spin = -self.bstring(spins[0]).count('1') + self.bstring(spins[1]).count('1')
                if spin != self.spin:
                    continue
            configs.append(config)
        return {i : j for i, j in enumerate(configs)}

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
            return self.U*self.bstring(sep1[0] & sep1[1]).count('1')
        else:
            # t-term
            H_t = 0

            # bitwise xor to identify differing elements
            xor = state2 ^ state1
            xor_str = self.bstring(xor, True)
            if ("11" in xor_str) and (sum(int(i) for i in xor_str) == 2):
                # ensure the differing indices are in a pair, and there is only one such pair
                if self.exclude_middle & xor != 0:
                    # make sure it's not the middle indices that got swapped
                    # since those don't indicate adjacent sites, but different spins on the first and last site
                    return -self.t
            if self.periodic:
                # final check to allow periodic boundary conditions
                alpha_hops = (xor & self.select_alpha) >> self.nsites
                beta_hops = xor & self.select_beta

                if ( beta_hops == self.periodic_select) and alpha_hops == 0 or \
                        (alpha_hops == self.periodic_select) and beta_hops == 0:
                    return -self.t
        return 0



    def build_matrix(self):
        # explicitly construct the Hamiltonian matrix

        if self._matrix is None:
            self._matrix = np.zeros((self.size, self.size))
            for i in range(self.size):
                state1 = self.index_table[i]
                for j in range(i+1):
                    state2 = self.index_table[j]
                    self._matrix[i,j] = self.calc_matrix_element(state1, state2)
                    self._matrix[j,i] = self._matrix[i,j]
        self.matrix = self._matrix
        self.check_sanity()
        return self._matrix

    def get_exact_energy(self):
        return np.linalg.eigh(self.build_matrix())



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    periodic=False

    H = Hubbard(nsites=4, t=1, U=2, nelec=4, periodic=periodic)

    assert H.calc_matrix_element(0b10100011, 0b01100011) == -1
    assert H.calc_matrix_element(0b11001010, 0b11001001) == -1
    assert H.calc_matrix_element(0b10011010, 0b10010110) == -1
    assert H.calc_matrix_element(0b10011010, 0b10010110) == -1
    assert H.calc_matrix_element(0b00111100, 0b00111010) == -1
    assert H.calc_matrix_element(0b10101001, 0b10011010) == 0
    assert H.calc_matrix_element(0b00111100, 0b00111100) == 0
    assert H.calc_matrix_element(0b10010110, 0b10011001) == 0
    assert H.calc_matrix_element(0b10001000, 0b10001001) == 0
    assert H.calc_matrix_element(0b11110110, 0b11110110) == 4
    assert H.calc_matrix_element(0b10101010, 0b00110011) == 0
    assert H.calc_matrix_element(0b10101010, 0b00111010) == -int(periodic)

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



