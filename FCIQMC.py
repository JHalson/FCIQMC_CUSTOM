import numpy as np
import ast
import system
import det_ops


def main(filename="FCIDUMP"):

    # Read in the Hamiltonian integrals from file
    sys_ham = det_ops.HAM(filename=filename, p_single=0.1)
    ref_energy = sys_ham.slater_condon(sys_ham.ref_det, sys_ham.ref_det, None, None)

    # Setup simulation parameters. See system.py for details.
    sim_params = system.PARAMS(totwalkers=400 , initwalkers=50, init_shift=0.1,
            shift_damp=0.5, timestep=2.e-2, det_thresh=0.75, eqm_iters=500,
            max_iter=1000, stats_cycle=5, seed=7, init_thresh=2.0)
    # Setup a statistics object, which accumulates various run-time variables.
    # See system.py for more details.
    sim_stats = system.STATS(sim_params, filename='fciqmc_stats', ref_energy=ref_energy)

    # Set up walker object as a dictionary.
    # Label determinants by the string representation of the list of occupied orbitals
    walkers = {repr(sys_ham.ref_det): sim_params.nwalk_init}
    sim_stats.nw = sim_params.nwalk_init

    for sim_stats.iter_curr in range(sim_params.max_iter):

        spawned_walkers = {}  # A dictionary to hold the spawned walkers of each iteration. With the initiator, this now takes a value which is a tuple of the weight, and the flag to indicate whether it came from an initiator determinant or not.
        sim_stats.nw = 0.0  # Recompute number of walkers each iteration
        sim_stats.ref_weight = 0.0
        sim_stats.nocc_dets = 0

        # Iterate over occupied (not all) determinants (keys) in the dictionary
        # Note that this is python3 format
        # Since we are modifying inplace, want to use .items, rather than setting up a true iterator
        for det_str, det_amp in list(walkers.items()):

            # Convert determinant string into a true list
            det = ast.literal_eval(det_str)

            # Accumulate current walker contribution to energy expectation values
            if det == sys_ham.ref_det:
                sim_stats.cycle_en_denom += det_amp
                sim_stats.ref_weight = det_amp
            else:
                # Find the parity and the excitation matrix between the determinant and the reference determinant
                excit_mat, parity = det_ops.calc_excit_mat_parity(sys_ham.ref_det, det)
                sim_stats.cycle_en_num += det_amp * sys_ham.slater_condon(sys_ham.ref_det, det, excit_mat, parity)

            # Stochastically round the walkers, if their amplitude is too low to ensure the walker list remains compact.
            if abs(det_amp) < sim_params.det_thresh:
                # Stochastically round up to sim_params.det_thresh with prob abs(det_amp)/sim_params.det_thresh, or disregard and skip this determinant
                if np.random.rand(1)[0] < abs(det_amp) / sim_params.det_thresh:
                    det_amp = sim_params.det_thresh * np.sign(det_amp)
                    # Also update it in the main walker list
                    walkers[det_str] = det_amp
                else:
                    # Kill walkers on this determinant entirely and remove the entry from the dictionary.
                    # Skip the rest of this walkers death/spawning
                    del walkers[det_str]
                    continue
            sim_stats.nw += abs(det_amp)
            sim_stats.nocc_dets += 1

            # Do a number of SPAWNING STEPS proportional to the modulus of the determinant amplitude
            nspawn = max(1, int(round(abs(det_amp))))
            for spawns in range(nspawn):

                # Generate determinant at random from determinant 'det'
                spawn_det, excit_mat, parity, p_gen = sys_ham.excit_gen(det)
                # Generate hamiltonian matrix element between these two determinants
                ham_el_spawn = sys_ham.slater_condon(det, spawn_det, excit_mat, parity)
                # Compute spawning probability
                p_spawn = -sim_params.timestep * ham_el_spawn * det_amp / (p_gen * nspawn)
                # Find the 'hashable' string representation of the determinant to look up in the spawned walker list
                spawn_str = repr(spawn_det)

                if abs(p_spawn) > 1.e-12:
                    if sim_params.init_thresh is None:
                        # Initiator approximation not is use. Set all spawns to be from initiators
                        init_flag = True
                    elif abs(det_amp) > sim_params.init_thresh:
                        # Parent amplitude above initiator theshold. Set init_flag to true.
                        init_flag = True
                    else:
                        # Parent is not sufficiently weighted to be an initiator. Mark it so it is only kept if spawning to an occupied determinant
                        init_flag = False
                    if spawn_str in spawned_walkers:
                        # If multiple spawning events to the same determinant, always mark the resulting spawned determinant as from an initiator
                        spawned_walkers[spawn_str][0] += p_spawn
                        spawned_walkers[spawn_str][1] = True
                    else:
                        spawned_walkers[spawn_str] = [p_spawn, init_flag]

            # DEATH STEP
            # Remember to now remove the reference energy from the determinant (this was done implicitly in part I)
            h_el_diag = sys_ham.slater_condon(det, det, None, None) - sim_stats.ref_energy
            walkers[det_str] -= sim_params.timestep * (h_el_diag - sim_params.shift) * det_amp

        # ANNIHILATION. Run through the list of newly spawned walkers, and merge with the main list.
        # However, if we are using the initiator approximation, we should also test whether we want
        # to transfer the walker weight across, or whether we want to abort the spawning attempt.
        for spawn_str, (spawn_amp, init_flag) in spawned_walkers.items():
            if spawn_str in walkers:
                # Merge with walkers already currently residing on this determinant
                walkers[spawn_str] += spawn_amp
            else:
                # Add as a new entry in the walker list (if it was marked as coming from an initiator determinant.
                if init_flag:
                    walkers[spawn_str] = spawn_amp

        # Every sim_params.stats_cycle iterations, readjust shift (if in variable shift mode) and print out statistics.
        if sim_stats.iter_curr % sim_params.stats_cycle == 0:
            # Update the shift, and turn on variable shift mode if enough walkers
            sim_params.update_shift(sim_stats)
            # Update the averaged statistics, and write out
            sim_stats.update_stats(sim_params)
            # subroutine for changing reference if necessary
            #change_ref(walkers, sys_ham, sim_stats)

    # Close the output file
    sim_stats.fout.close()

def change_ref(walkers=None, sys_ham=None, sim_stats=None, det=None, tol=0.5):
    ref_changed = False
    if det is not None:
        # change to the given reference
        new_ref = repr(det)
        ref_changed = True
    else:
        # change to the next most occupied determinant
        new_ref = repr(sys_ham.ref_det)
        for label, amp in walkers.items():
            if abs(amp / walkers[new_ref]) > (1 + tol):
                new_ref = label
                ref_changed = True
                tol = 0
    if ref_changed:
        print('changing reference from {} to {}'.format(sys_ham.ref_det, new_ref))
        sys_ham.ref_det = ast.literal_eval(new_ref)
        ref_energy = sys_ham.slater_condon(sys_ham.ref_det, sys_ham.ref_det, None, None)
        sim_stats.ref_energy = ref_energy
    return ref_changed


if __name__ == "__main__":
    main("TEST.FCIDUMP")


