'''
Solve FCI problem with given 1-electron and 2-electron Hubbard Hamiltonian
'''

import numpy as np
from pyscf import fci, gto, scf, ao2mo

norb = 4
n_alpha = norb // 2
n_beta = norb // 2
nelec = n_alpha + n_beta
U = 2.0

# Define 1D Hubbard hamiltonian
h1 = np.zeros((norb,norb))
for i in range(norb-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
#h1[norb-1,0] = h1[0,norb-1] = -1.0  # Periodic boundary conditions
eri = np.zeros((norb,norb,norb,norb))
for i in range(norb):
    eri[i,i,i,i] = U

#
# Generally, direct_spin1.kernel is the FCI object which can handle all generic systems.
#
e, fcivec = fci.direct_spin1.kernel(h1, eri, norb, nelec, verbose=6)
print('FCI Energy is {}'.format(e))

# Alternative options below:
#
# A better way is to create a FCI (=FCISolver) object because FCI object offers
# more options to control the calculation.
#
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
e, fcivec = cisolver.kernel(h1, eri, norb, (n_alpha, n_beta), verbose=5)  # n_alpha alpha, n_beta beta electrons
print('FCI Energy is {}'.format(e))

#
# If you are sure the system ground state is singlet, you can use spin0 solver.
# Spin symmetry is considered in spin0 solver to reduce cimputation cost.
#
cisolver = fci.direct_spin0.FCI()
cisolver.verbose = 5
e, fcivec = cisolver.kernel(h1, eri, norb, nelec)
print('FCI Energy is {}'.format(e))

# OPTIONAL: Perform a mean-field hartree-fock calculation, by overwriting some internal objects
mol = gto.M(verbose=3)
mol.nelectron = nelec
# Setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
# attribute) to be used in the post-HF calculations.  Without this parameter,
# some post-HF method (particularly in the MO integral transformation) may
# ignore the customized Hamiltonian if memory is not enough.
mol.incore_anyway = True
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(norb)
mf._eri = ao2mo.restore(8, eri, norb)
mf.kernel()


from Hubbard import Hubbard

H = Hubbard(nsites=norb, t=1, U=2, nelec=norb)

H.build_matrix()
with np.printoptions(threshold=H.size**2, linewidth=136):
    print(H.matrix.shape)
    print(H.matrix)


print("ED Energy = {}".format(H.get_exact_energy()[0][0]))