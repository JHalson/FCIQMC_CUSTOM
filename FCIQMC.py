import numpy as np
import ast
import system
import det_ops


class FCIQMC:
    def __init__(self, ham_filename, **kwargs):
        ham_opts, param_opts, misc_opts = self.set_defaults(kwargs)

        # Read in the Hamiltonian integrals from file
        self.sys_ham = det_ops.HAM(filename=ham_filename, **ham_opts)
        ref_energy = self.sys_ham.slater_condon(self.sys_ham.ref_det, self.sys_ham.ref_det, None, None)

        # Setup simulation parameters. See system.py for details.
        self.sim_params = system.PARAMS(**param_opts)

        # Setup a statistics object, which accumulates various run-time variables.
        # See system.py for more details.
        self.sim_stats = system.STATS(self.sim_params, filename='fciqmc_stats', ref_energy=ref_energy)

        # Set up walker object as a dictionary.
        # Label determinants by the string representation of the list of occupied orbitals
        self.walkers = {repr(self.sys_ham.ref_det): self.sim_params.nwalk_init}
        self.sim_stats.nw = self.sim_params.nwalk_init

    def set_defaults(self, kwargs_dict):
        p_single = kwargs_dict.pop('p_single') if 'p_single' in kwargs_dict else 0.1
        self.use_hubbard = kwargs_dict.pop('use_hubbard') if 'use_hubbard' in kwargs_dict else False
        if self.use_hubbard:
            p_single = 1
            kwargs_dict['init_thresh'] = None

        ham_opts = dict(p_single=p_single)

        # arguments for the PARAMS object
        param_opts = dict(totwalkers=1000, initwalkers=50, init_shift=0.1,
                        shift_damp=0.5, timestep=2.e-2, det_thresh=0.75, eqm_iters=500,
                        max_iter=2000, stats_cycle=5, seed=7, init_thresh=2.0)
        param_opts.update(kwargs_dict)

        misc_opts = dict()

        return ham_opts, param_opts, misc_opts


    def kernel(self):

        for self.sim_stats.iter_curr in range(self.sim_params.max_iter):

            spawned_walkers = {}  # A dictionary to hold the spawned walkers of each iteration. With the initiator, this now takes a value which is a tuple of the weight, and the flag to indicate whether it came from an initiator determinant or not.
            self.sim_stats.nw = 0.0  # Recompute number of walkers each iteration
            self.sim_stats.ref_weight = 0.0
            self.sim_stats.nocc_dets = 0

            # Iterate over occupied (not all) determinants (keys) in the dictionary
            # Note that this is python3 format
            # Since we are modifying inplace, want to use .items, rather than setting up a true iterator
            for det_str, det_amp in list(self.walkers.items()):

                # Convert determinant string into a true list
                det = ast.literal_eval(det_str)

                # Accumulate current walker contribution to energy expectation values
                if det == self.sys_ham.ref_det:
                    self.sim_stats.cycle_en_denom += det_amp
                    self.sim_stats.ref_weight = det_amp
                else:
                    # Find the parity and the excitation matrix between the determinant and the reference determinant
                    excit_mat, parity = det_ops.calc_excit_mat_parity(self.sys_ham.ref_det, det)
                    self.sim_stats.cycle_en_num += det_amp * self.sys_ham.slater_condon(self.sys_ham.ref_det, det, excit_mat, parity)

                # Stochastically round the walkers, if their amplitude is too low to ensure the walker list remains compact.
                if abs(det_amp) < self.sim_params.det_thresh:
                    # Stochastically round up to sim_params.det_thresh with prob abs(det_amp)/sim_params.det_thresh, or disregard and skip this determinant
                    if np.random.rand(1)[0] < abs(det_amp) / self.sim_params.det_thresh:
                        det_amp = self.sim_params.det_thresh * np.sign(det_amp)
                        # Also update it in the main walker list
                        self.walkers[det_str] = det_amp
                    else:
                        # Kill walkers on this determinant entirely and remove the entry from the dictionary.
                        # Skip the rest of this walkers death/spawning
                        del self.walkers[det_str]
                        continue
                self.sim_stats.nw += abs(det_amp)
                self.sim_stats.nocc_dets += 1

                # Do a number of SPAWNING STEPS proportional to the modulus of the determinant amplitude
                nspawn = max(1, int(round(abs(det_amp))))
                for spawns in range(nspawn):

                    # Generate determinant at random from determinant 'det'
                    spawn_det, excit_mat, parity, p_gen = self.sys_ham.excit_gen(det)
                    # Generate hamiltonian matrix element between these two determinants
                    ham_el_spawn = self.sys_ham.slater_condon(det, spawn_det, excit_mat, parity)
                    # Compute spawning probability
                    p_spawn = -self.sim_params.timestep * ham_el_spawn * det_amp / (p_gen * nspawn)
                    # Find the 'hashable' string representation of the determinant to look up in the spawned walker list
                    spawn_str = repr(spawn_det)

                    if abs(p_spawn) > 1.e-12:
                        if self.sim_params.init_thresh is None:
                            # Initiator approximation not is use. Set all spawns to be from initiators
                            init_flag = True
                        elif abs(det_amp) > self.sim_params.init_thresh:
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
                h_el_diag = self.sys_ham.slater_condon(det, det, None, None) - self.sim_stats.ref_energy
                self.walkers[det_str] -= self.sim_params.timestep * (h_el_diag - self.sim_params.shift) * det_amp

            # ANNIHILATION. Run through the list of newly spawned walkers, and merge with the main list.
            # However, if we are using the initiator approximation, we should also test whether we want
            # to transfer the walker weight across, or whether we want to abort the spawning attempt.
            for spawn_str, (spawn_amp, init_flag) in spawned_walkers.items():
                if spawn_str in self.walkers:
                    if self.use_hubbard:
                        # 1D Hubbard should not have annihilations between opposite signed walkers
                        assert np.sign(self.walkers[spawn_str]) == np.sign(spawn_amp)
                    # Merge with walkers already currently residing on this determinant
                    self.walkers[spawn_str] += spawn_amp
                else:
                    # Add as a new entry in the walker list (if it was marked as coming from an initiator determinant.
                    if init_flag:
                        self.walkers[spawn_str] = spawn_amp

            # Every sim_params.stats_cycle iterations, readjust shift (if in variable shift mode) and print out statistics.
            if self.sim_stats.iter_curr % self.sim_params.stats_cycle == 0:
                # Update the shift, and turn on variable shift mode if enough walkers
                self.sim_params.update_shift(self.sim_stats)
                # Update the averaged statistics, and write out
                self.sim_stats.update_stats(self.sim_params)
                # subroutine for changing reference if necessary
                self.change_ref()

        # Close the output file
        self.sim_stats.fout.close()

    def change_ref(self, det=None, tol=0.5):
        ref_changed = False
        if det is not None:
            # change to the given reference
            new_ref = repr(det)
            ref_changed = True
        else:
            # change to the next most occupied determinant
            new_ref = repr(self.sys_ham.ref_det)
            for label, amp in self.walkers.items():
                if abs(amp / self.walkers[new_ref]) > (1 + tol):
                    new_ref = label
                    ref_changed = True
                    tol = 0
        if ref_changed:
            print('changing reference from {} to {}'.format(self.sys_ham.ref_det, new_ref))
            self.sys_ham.ref_det = ast.literal_eval(new_ref)
            ref_energy = self.sys_ham.slater_condon(self.sys_ham.ref_det, self.sys_ham.ref_det, None, None)
            self.sim_stats.ref_energy = ref_energy
        return ref_changed


if __name__ == "__main__":
    F = FCIQMC('TEST.FCIDUMP')
    F.kernel()


