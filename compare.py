'''
Solve FCI problem with given 1-electron and 2-electron Hubbard Hamiltonian
'''

import numpy
from pyscf import fci, gto, scf, ao2mo
from Hubbard import Hubbard

def test_pyscf(U=2, norb=4):
    nelec = norb
    n_beta = norb // 2
    n_alpha = nelec - n_beta

    # Define 1D Hubbard hamiltonian
    h1 = numpy.zeros((norb,norb))
    for i in range(norb-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    #h1[norb-1,0] = h1[0,norb-1] = -1.0  # Periodic boundary conditions
    eri = numpy.zeros((norb,norb,norb,norb))
    for i in range(norb):
        eri[i,i,i,i] = U

    from pyscf.tools.fcidump import from_integrals
    from_integrals('FCIDUMP.PySCF', h1, eri, norb, nelec)

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
    mf.get_ovlp = lambda *args: numpy.eye(norb)
    mf._eri = ao2mo.restore(8, eri, norb)
    e = mf.kernel()
    print('FCI Energy is {}'.format(e))

def test_exact_diag(U=2, norb=4, spin=None):
    H = Hubbard(nsites=norb, t=1, U=U, nelec=norb, spin=spin)
    return H.get_exact_energy()[0][0], H.matrix

def test_fciqmc(U=2, norb=4):
    from FCIQMC import main
    write_fcidump(U, norb)
    main()

def write_fcidump(U, norb, nelec=None, filename="FCIDUMP"):
    if not nelec: nelec = norb
    # write an FCIDUMP for a Hubbard model
    with open(filename, "w") as f:
        # make header
        f.write("&FCI NORB={}, NELEC={}, MS2=0\n".format(norb, nelec))
        f.write(" ORBSYM="+"1,"*norb+"\n")
        f.write(" ISYM=1\n")
        f.write("&END\n")

        f.write("0 0 0 0 0\n")

        for i in range(1, norb+1):
            f.write("{} {} {} {} {}\n".format(U, i, i, i, i))
        for i in range(1,norb):
            f.write("{} {} {} 0 0 \n".format(-1, i + 1, i))

def test_neci(U=2, norb=4):
    from subprocess import Popen, PIPE
    write_fcidump(U, norb)
    Popen("rm FCIMCStats*", cwd="./neci_workdir", shell=True).wait()
    Popen("sed 's/<nelec>/{}/' neci.inp.template > neci.inp".format(norb), cwd="./neci_workdir", shell=True).wait()
    resp = Popen(("./neci", "neci.inp"), stdout=PIPE, stderr=PIPE, cwd="./neci_workdir").communicate()[0].decode('UTF-8')
    for line in resp.split('\n'):
        if line.strip().startswith("Final energy estimate for state 1:"):
            return float(line.split(":")[1].strip())
    return None

def gen_hamiltonian_from_slater_condon(U=2, norb=4):
    from itertools import combinations, product
    from det_ops import HAM, calc_excit_mat_parity
    import numpy as np
    write_fcidump(U, norb)
    H = HAM(filename="FCIDUMP", p_single=0.1)

    alpha = [list(k) for k in combinations(list(range(0,H.nelec)), H.nelec//2)]
    beta = [list(k) for k in combinations(list(range(H.nelec, 2*H.nelec)), H.nelec//2)]
    dets = [el[0] + el[1] for el in product(alpha, beta, repeat=1)]

    ham = np.zeros((len(dets), len(dets)))
    for i, e1 in enumerate(dets):
        for j, e2 in enumerate(dets):
            ham[i,j] = H.slater_condon(e1, e2, *calc_excit_mat_parity(e1,e2))
    return ham

import numpy as np

U=2
norb = 6

H = gen_hamiltonian_from_slater_condon(U, norb)
assert(np.allclose(H.T, H))
print("energy of matrix generated from Slater Condon: {}".format(np.linalg.eigh(H)[0][0]))
print("")


test_pyscf(U=U, norb=norb)
ED_e, ED_H = test_exact_diag(U=U, norb=norb, spin=norb%2)


print("Energy from ED: {}".format(ED_e))

e_neci = test_neci(U=U, norb=norb)
print("Energy from NECI: {}".format(e_neci))
test_fciqmc(U=U, norb=norb)

