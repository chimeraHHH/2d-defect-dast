import warnings
from math import sqrt, pi
import numpy as np

from ase.units import Ha
from gpaw import BadParallelization
from gpaw.mpi import world
from gpaw.density import redistribute_array, redistribute_atomic_matrices
from gpaw.sphere.lebedev import weight_n
from gpaw.utilities import (pack_atomic_matrices, pack_density,
                            unpack_atomic_matrices)
from gpaw.xc.gllb import safe_sqr
from gpaw.xc.gllb.contribution import Contribution

# XXX Work in process
debug = False


def d(*args):
    if debug:
        print(args)


class ResponsePotential:
    """Container for response potential"""
    def __init__(self, response, vt_sG, vt_sg, D_asp, Dresp_asp):
        self.response = response
        self.vt_sG = vt_sG
        self.vt_sg = vt_sg
        self.D_asp = D_asp
        self.Dresp_asp = Dresp_asp

    def redistribute(self, new_response):
        old_response = self.response
        new_wfs = new_response.wfs
        new_density = new_response.density
        self.vt_sG = redistribute_array(self.vt_sG,
                                        old_response.density.gd,
                                        new_density.gd,
                                        new_wfs.nspins,
                                        new_wfs.kptband_comm)
        if self.vt_sg is not None:
            self.vt_sg = redistribute_array(self.vt_sg,
                                            old_response.density.finegd,
                                            new_density.finegd,
                                            new_wfs.nspins,
                                            new_wfs.kptband_comm)

        def redist_asp(D_asp):
            return redistribute_atomic_matrices(D_asp,
                                                new_density.gd,
                                                new_wfs.nspins,
                                                new_density.setups,
                                                new_density.atom_partition,
                                                new_wfs.kptband_comm)

        self.D_asp = redist_asp(self.D_asp)
        self.Dresp_asp = redist_asp(self.Dresp_asp)
        self.response = new_response


class C_Response(Contribution):
    """Response contribution for GLLB functionals.

    Parameters
    ----------
    weight
        Weight of the contribution
    coefficients
        Coefficient calculator object
    metallic
        If True, then Fermi level is used as the reference
        energy for coefficients instead of the HOMO energy.
        This is necessary to get sensible results in metallic systems.
    damp
        Small value to damp divisions by zero
    """
    def __init__(self,
                 weight: float,
                 coefficients, *,
                 metallic: bool = False,
                 damp: float = 1e-10):
        Contribution.__init__(self, weight)
        d('In c_Response __init__', self)
        self.coefficients = coefficients
        self.vt_sg = None
        self.vt_sG = None
        self.D_asp = None
        self.Dresp_asp = None
        self.Ddist_asp = None
        self.Drespdist_asp = None
        self.damp = damp
        self.fix_potential = False
        self.metallic = metallic

        # For logging reference energy
        self.eref_s = None
        self.eref_source_s = None

    def set_damp(self, damp):
        self.damp = damp

    def get_name(self):
        return 'RESPONSE'

    def get_desc(self):
        desc = []
        if self.metallic:
            desc += ['metallic']
        desc += [self.coefficients.get_description()]
        return ', '.join(desc)

    def initialize(self, density, hamiltonian, wfs):
        Contribution.initialize(self, density, hamiltonian, wfs)
        self.coefficients.initialize(wfs)
        if self.Dresp_asp is None:
            assert self.density.D_asp is None
        # XXX With the deprecated `fixdensity=True` option this
        # initialize() function is called both before AND after read()!
        # Thus, the second call of initialize() would discard the read
        # potential unless we check if it was already allocated.
        # Remove this if statement once `fixdensity=True` option has
        # been removed from the calculator.
        if self.vt_sg is None:
            self.vt_sG = self.gd.empty(self.nspins)
            self.vt_sg = self.finegd.empty(self.nspins)

    def initialize_1d(self, ae):
        Contribution.initialize_1d(self, ae)
        self.coefficients.initialize_1d(ae)

    def initialize_from_other_response(self, response):
        pot = ResponsePotential(response,
                                response.vt_sG,
                                response.vt_sg,
                                response.D_asp,
                                response.Dresp_asp)
        pot.redistribute(self)
        self.vt_sG = pot.vt_sG
        self.vt_sg = pot.vt_sg
        self.D_asp = pot.D_asp
        self.Dresp_asp = pot.Dresp_asp
        self.Ddist_asp = self.distribute_D_asp(pot.D_asp)
        self.Drespdist_asp = self.distribute_D_asp(pot.Dresp_asp)
        self.eref_s = response.eref_s
        self.eref_source_s = response.eref_source_s

    # Calcualte the GLLB potential and energy 1d
    def add_xc_potential_and_energy_1d(self, v_g):
        w_i = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g += self.weight * np.dot(w_i, u2_j) / (
            np.dot(self.ae.f_j, u2_j) + self.damp)
        return 0.0  # Response part does not contribute to energy

    def update_potentials(self):
        d('In update response potential')
        if self.fix_potential:
            # Skip the evaluation of the potential so that
            # the existing potential is used. This is relevant for
            # band structure calculation so that the potential
            # does not get updated with the other k-point sampling.
            return

        if self.wfs.kpt_u[0].eps_n is None:
            # This skips the evaluation of the potential so that
            # the existing one is used.
            # This happens typically after reading when occupations
            # haven't been calculated yet and the potential read earlier
            # is used.
            return

        # Calculate reference energy
        self.eref_s = []
        self.eref_source_s = []
        if self.metallic:
            # Use Fermi level as reference levels
            fermilevel = self.wfs.fermi_level
            assert isinstance(fermilevel, float), \
                'GLLBSCM supports only a single Fermi level'
            for s in range(self.wfs.nspins):
                self.eref_source_s.append('Fermi level')
                self.eref_s.append(fermilevel)
        else:
            # Find homo and lumo levels for each spin
            for s in range(self.wfs.nspins):
                homo, lumo = self.wfs.get_homo_lumo(s, _gllb=True)
                # Check that homo and lumo are reasonable
                if homo > lumo:
                    # HOMO higher than LUMO; set Fermi level as reference
                    fermilevel = self.wfs.fermi_level
                    self.eref_source_s.append('Fermi level')
                    self.eref_s.append(fermilevel)
                else:
                    self.eref_source_s.append('HOMO')
                    self.eref_s.append(homo)

        w_kn = self.coefficients.get_coefficients(self.wfs.kpt_u,
                                                  eref_s=self.eref_s)
        f_kn = [kpt.f_n for kpt in self.wfs.kpt_u]
        if w_kn is not None:
            self.vt_sG[:] = 0.0
            nt_sG = self.gd.zeros(self.nspins)

            for kpt, w_n in zip(self.wfs.kpt_u, w_kn):
                self.wfs.add_to_density_from_k_point_with_occupation(
                    self.vt_sG, kpt, w_n)
                self.wfs.add_to_density_from_k_point(nt_sG, kpt)

            self.wfs.kptband_comm.sum(nt_sG)
            self.wfs.kptband_comm.sum(self.vt_sG)

            if self.wfs.kd.symmetry:
                for nt_G, vt_G in zip(nt_sG, self.vt_sG):
                    self.wfs.kd.symmetry.symmetrize(nt_G, self.gd)
                    self.wfs.kd.symmetry.symmetrize(vt_G, self.gd)

            d('response update D_asp', world.rank, self.Dresp_asp.keys(),
              self.D_asp.keys())
            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.Dresp_asp, w_kn)
            self.Drespdist_asp = self.distribute_D_asp(self.Dresp_asp)
            d('response update Drespdist_asp', world.rank,
              self.Dresp_asp.keys(), self.D_asp.keys())
            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.D_asp, f_kn)
            self.Ddist_asp = self.distribute_D_asp(self.D_asp)

            self.vt_sG /= nt_sG + self.damp

        self.density.distribute_and_interpolate(self.vt_sG, self.vt_sg)

    def calculate(self, e_g, n_sg, v_sg):
        self.update_potentials()
        v_sg += self.weight * self.vt_sg

    def distribute_D_asp(self, D_asp):
        return self.density.atomdist.to_work(D_asp)

    def calculate_energy_and_derivatives(self, setup, D_sp, H_sp, a,
                                         addcoredensity=True):
        # Get the XC-correction instance
        c = setup.xc_correction
        ncresp_g = setup.extra_xc_data['core_response'] / self.nspins
        if not addcoredensity:
            ncresp_g[:] = 0.0

        for D_p, dEdD_p, Dresp_p in zip(self.Ddist_asp[a], H_sp,
                                        self.Drespdist_asp[a]):
            D_Lq = np.dot(c.B_pqL.T, D_p)
            n_Lg = np.dot(D_Lq, c.n_qg)  # Construct density
            if addcoredensity:
                n_Lg[0] += c.nc_g * sqrt(4 * pi) / self.nspins
            nt_Lg = np.dot(
                D_Lq, c.nt_qg
            )  # Construct smooth density (_without smooth core density_)

            Dresp_Lq = np.dot(c.B_pqL.T, Dresp_p)
            nresp_Lg = np.dot(Dresp_Lq, c.n_qg)  # Construct 'response density'
            nrespt_Lg = np.dot(
                Dresp_Lq, c.nt_qg
            )  # Construct smooth 'response density' (w/o smooth core)

            for w, Y_L in zip(weight_n, c.Y_nL):
                nt_g = np.dot(Y_L, nt_Lg)
                nrespt_g = np.dot(Y_L, nrespt_Lg)
                x_g = nrespt_g / (nt_g + self.damp)
                dEdD_p -= self.weight * w * np.dot(
                    np.dot(c.B_pqL, Y_L), np.dot(c.nt_qg, x_g * c.rgd.dv_g))

                n_g = np.dot(Y_L, n_Lg)
                nresp_g = np.dot(Y_L, nresp_Lg)
                x_g = (nresp_g + ncresp_g) / (n_g + self.damp)

                dEdD_p += self.weight * w * np.dot(
                    np.dot(c.B_pqL, Y_L), np.dot(c.n_qg, x_g * c.rgd.dv_g))

        return 0.0

    def integrate_sphere(self, a, Dresp_sp, D_sp, Dwf_p, spin=0):
        c = self.wfs.setups[a].xc_correction
        Dresp_p, D_p = Dresp_sp[spin], D_sp[spin]
        D_Lq = np.dot(c.B_pqL.T, D_p)
        n_Lg = np.dot(D_Lq, c.n_qg)  # Construct density
        n_Lg[0] += c.nc_g * sqrt(4 * pi) / len(D_sp)
        nt_Lg = np.dot(D_Lq, c.nt_qg
                       )  # Construct smooth density (without smooth core)
        Dresp_Lq = np.dot(c.B_pqL.T, Dresp_p)  # Construct response
        nresp_Lg = np.dot(Dresp_Lq, c.n_qg)  # Construct 'response density'
        nrespt_Lg = np.dot(
            Dresp_Lq, c.nt_qg
        )  # Construct smooth 'response density' (w/o smooth core)
        Dwf_Lq = np.dot(c.B_pqL.T, Dwf_p)  # Construct lumo wf
        nwf_Lg = np.dot(Dwf_Lq, c.n_qg)
        nwft_Lg = np.dot(Dwf_Lq, c.nt_qg)
        E = 0.0
        for w, Y_L in zip(weight_n, c.Y_nL):
            v = np.dot(Y_L, nwft_Lg) * np.dot(Y_L, nrespt_Lg) / (
                np.dot(Y_L, nt_Lg) + self.damp)
            E -= self.weight * w * np.dot(v, c.rgd.dv_g)
            v = np.dot(Y_L, nwf_Lg) * np.dot(Y_L, nresp_Lg) / (
                np.dot(Y_L, n_Lg) + self.damp)
            E += self.weight * w * np.dot(v, c.rgd.dv_g)
        return E

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        v_g = np.zeros(self.ae.N)
        n_g = np.zeros(self.ae.N)
        for w_n, f_n, u_n in zip(w_ln, self.ae.f_ln,
                                 self.ae.s_ln):  # For each angular momentum
            u2_n = safe_sqr(u_n)
            v_g += np.dot(w_n, u2_n)
            n_g += np.dot(f_n, u2_n)

        vt_g += self.weight * v_g / (n_g + self.damp)
        return 0.0  # Response part does not contribute to energy

    def calculate_delta_xc(self, homolumo=None):
        warnings.warn(
            'The function calculate_delta_xc() is deprecated. '
            'Please use calculate_discontinuity_potential() instead. '
            'See documentation on calculating band gap with GLLBSC.',
            DeprecationWarning)

        if homolumo is None:
            # Calculate band gap

            # This always happens, so we don't warn.
            # We should perhaps print it as an ordinary message,
            # but we do not have a file here to which to print.
            # import warnings
            # warnings.warn('Calculating KS-gap directly from the k-points, '
            #               'can be inaccurate.')
            homolumo = self.wfs.get_homo_lumo(_gllb=True)
        homo, lumo = homolumo
        dxc_pot = self.calculate_discontinuity_potential(homo * Ha, lumo * Ha)
        self.Dxc_vt_sG = dxc_pot.vt_sG
        self.Dxc_D_asp = dxc_pot.D_asp
        self.Dxc_Dresp_asp = dxc_pot.Dresp_asp

    def calculate_discontinuity_potential(self, homo, lumo):
        r"""Calculate the derivative discontinuity potential.

        This function calculates $`\Delta_{xc}(r)`$ as given by
        Eq. (24) of https://doi.org/10.1103/PhysRevB.82.115106

        Parameters:

        homo:
            homo energy (or energies in spin-polarized case) in eV
        lumo:
            lumo energy (or energies in spin-polarized case) in eV

        Returns:

        dxc_pot: Discontinuity potential
        """
        if self.nspins == 2:
            eref_s = np.asarray(homo)
            eref_lumo_s = np.asarray(lumo)
        else:
            eref_s = np.asarray([homo])
            eref_lumo_s = np.asarray([lumo])
        assert len(eref_s) == len(eref_lumo_s) == self.nspins
        eref_s /= Ha
        eref_lumo_s /= Ha

        dxc_Dresp_asp = self.empty_atomic_matrix()
        dxc_D_asp = self.empty_atomic_matrix()
        for a in self.density.D_asp:
            ni = self.wfs.setups[a].ni
            dxc_Dresp_asp[a] = np.zeros((self.nspins, ni * (ni + 1) // 2))
            dxc_D_asp[a] = np.zeros((self.nspins, ni * (ni + 1) // 2))

        # Calculate new response potential with LUMO reference
        w_kn = self.coefficients.get_coefficients_for_lumo_perturbation(
            self.wfs.kpt_u, eref_s=eref_s, eref_lumo_s=eref_lumo_s)
        f_kn = [kpt.f_n for kpt in self.wfs.kpt_u]

        dxc_vt_sG = self.gd.zeros(self.nspins)
        nt_sG = self.gd.zeros(self.nspins)
        for kpt, w_n in zip(self.wfs.kpt_u, w_kn):
            self.wfs.add_to_density_from_k_point_with_occupation(dxc_vt_sG,
                                                                 kpt, w_n)
            self.wfs.add_to_density_from_k_point(nt_sG, kpt)

        self.wfs.kptband_comm.sum(nt_sG)
        self.wfs.kptband_comm.sum(dxc_vt_sG)

        if self.wfs.kd.symmetry:
            for nt_G, dxc_vt_G in zip(nt_sG, dxc_vt_sG):
                self.wfs.kd.symmetry.symmetrize(nt_G, self.gd)
                self.wfs.kd.symmetry.symmetrize(dxc_vt_G, self.gd)

        dxc_vt_sG /= nt_sG + self.damp

        self.wfs.calculate_atomic_density_matrices_with_occupation(
            dxc_Dresp_asp, w_kn)
        self.wfs.calculate_atomic_density_matrices_with_occupation(
            dxc_D_asp, f_kn)
        dxc_pot = ResponsePotential(self, dxc_vt_sG, None,
                                    dxc_D_asp, dxc_Dresp_asp)
        return dxc_pot

    def calculate_delta_xc_perturbation(self):
        warnings.warn(
            'The function calculate_delta_xc_perturbation() is deprecated. '
            'Please use calculate_discontinuity() instead. '
            'See documentation on calculating band gap with GLLBSC.',
            DeprecationWarning)
        dxc_pot = ResponsePotential(self, self.Dxc_vt_sG, None, self.Dxc_D_asp,
                                    self.Dxc_Dresp_asp)
        if self.nspins != 1:
            ret = []
            for spin in range(self.nspins):
                ret.append(self.calculate_discontinuity(dxc_pot, spin=spin))
            return ret
        return self.calculate_discontinuity(dxc_pot)

    def calculate_delta_xc_perturbation_spin(self, s=0):
        warnings.warn(
            'The function calculate_delta_xc_perturbation_spin() is '
            'deprecated. Please use calculate_discontinuity_spin() instead. '
            'See documentation on calculating band gap with GLLBSC.',
            DeprecationWarning)
        dxc_pot = ResponsePotential(self, self.Dxc_vt_sG, None, self.Dxc_D_asp,
                                    self.Dxc_Dresp_asp)
        return self.calculate_discontinuity(dxc_pot, spin=s)

    def calculate_discontinuity(self,
                                dxc_pot: ResponsePotential,
                                spin: int = None):
        r"""Calculate the derivative discontinuity.

        This function evaluates the perturbation theory expression
        for the derivative discontinuity value as given by
        Eq. (25) of https://doi.org/10.1103/PhysRevB.82.115106

        Parameters:

        dxc_pot:
            Discontinuity potential.
        spin:
            Spin value.

        Returns:

        KSgap:
            Kohn-Sham gap in eV
        dxc:
            Derivative discontinuity in eV
        """
        if spin is None:
            if self.nspins != 1:
                raise ValueError('Specify spin for the discontinuity.')
            spin = 0

        # Redistribute discontinuity potential
        if dxc_pot.response is not self:
            dxc_pot.redistribute(self)

        homo, lumo = self.wfs.get_homo_lumo(spin, _gllb=True)
        KSgap = lumo - homo

        # Find the lumo-orbital of this spin
        if self.wfs.bd.comm.size != 1:
            raise BadParallelization(
                'Band parallelization is not supported by '
                'calculate_discontinuity().')
        eps_n = self.wfs.kpt_qs[0][spin].eps_n
        lumo_n = (eps_n < self.wfs.fermi_level).sum()

        # Calculate the expectation value for all k points, and keep
        # the smallest energy value
        nt_G = self.gd.empty()
        min_energy = np.inf
        for u, kpt in enumerate(self.wfs.kpt_u):
            if kpt.s == spin:
                nt_G[:] = 0.0
                self.wfs.add_orbital_density(nt_G, kpt, lumo_n)
                E = 0.0
                for a in dxc_pot.D_asp:
                    D_sp = dxc_pot.D_asp[a]
                    Dresp_sp = dxc_pot.Dresp_asp[a]
                    P_ni = kpt.P_ani[a]
                    Dwf_p = pack_density(
                        np.outer(P_ni[lumo_n].T.conj(), P_ni[lumo_n]).real)
                    E += self.integrate_sphere(a, Dresp_sp, D_sp, Dwf_p,
                                               spin=spin)
                E = self.gd.comm.sum_scalar(E)
                E += self.gd.integrate(nt_G * dxc_pot.vt_sG[spin])
                E += kpt.eps_n[lumo_n] - lumo
                min_energy = min(min_energy, E)

        # Take the smallest value over all distributed k points
        dxc = self.wfs.kd.comm.min_scalar(min_energy)
        return KSgap * Ha, dxc * Ha

    def calculate_discontinuity_from_average(self,
                                             dxc_pot: ResponsePotential,
                                             spin: int,
                                             testing: bool = False):
        msg = ('This function estimates discontinuity by calculating '
               'the average of the discontinuity potential. '
               'Use only for testing and debugging.')
        if not testing:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)
        assert self.wfs.world.size == 1

        # Calculate average of lumo reference response potential
        dxc = np.average(dxc_pot.vt_sG[spin])
        return dxc * Ha

    def initialize_from_atomic_orbitals(self, basis_functions):
        # Initialize 'response-density' and density-matrices
        # print('Initializing from atomic orbitals')
        self.Dresp_asp = self.empty_atomic_matrix()
        setups = self.wfs.setups

        for a in self.density.D_asp.keys():
            ni = setups[a].ni
            self.Dresp_asp[a] = np.zeros((self.nspins, ni * (ni + 1) // 2))

        self.D_asp = self.empty_atomic_matrix()
        f_asi = {}
        w_asi = {}

        for a in basis_functions.atom_indices:
            if setups[a].type == 'ghost':
                w_j = [0.0]
            else:
                w_j = setups[a].extra_xc_data['w_j']

            # Basis function coefficients based of response weights
            w_si = setups[a].calculate_initial_occupation_numbers(
                0, False,
                charge=0,
                nspins=self.nspins,
                f_j=w_j)
            # Basis function coefficients based on density
            f_si = setups[a].calculate_initial_occupation_numbers(
                0, False,
                charge=0,
                nspins=self.nspins)

            if a in basis_functions.my_atom_indices:
                self.Dresp_asp[a] = setups[a].initialize_density_matrix(w_si)
                self.D_asp[a] = setups[a].initialize_density_matrix(f_si)

            f_asi[a] = f_si
            w_asi[a] = w_si

        self.Drespdist_asp = self.distribute_D_asp(self.Dresp_asp)
        self.Ddist_asp = self.distribute_D_asp(self.D_asp)
        nt_sG = self.gd.zeros(self.nspins)
        basis_functions.add_to_density(nt_sG, f_asi)
        self.vt_sG = self.gd.zeros(self.nspins)
        basis_functions.add_to_density(self.vt_sG, w_asi)
        # Update vt_sG to correspond atomic response potential. This will be
        # used until occupations and eigenvalues are available.
        self.vt_sG /= nt_sG + self.damp
        self.vt_sg = self.finegd.zeros(self.nspins)
        self.density.distribute_and_interpolate(self.vt_sG, self.vt_sg)

    def get_extra_setup_data(self, extra_data):
        ae = self.ae
        njcore = ae.njcore
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        extra_data['w_ln'] = w_ln

        w_j = self.coefficients.get_coefficients_1d()
        x_g = np.dot(w_j[:njcore], safe_sqr(ae.u_j[:njcore]))
        x_g[1:] /= ae.r[1:] ** 2 * 4 * np.pi
        x_g[0] = x_g[1]
        extra_data['core_response'] = x_g

    def write(self, writer):
        """Writes response specific data."""
        wfs = self.wfs
        kpt_comm = wfs.kd.comm
        gd = wfs.gd

        # Write the pseudodensity on the coarse grid:
        shape = (wfs.nspins,) + tuple(gd.get_size_of_global_array())
        writer.add_array('gllb_pseudo_response_potential', shape)
        if kpt_comm.rank == 0:
            for vt_G in self.vt_sG:
                writer.fill(gd.collect(vt_G) * Ha)

        writer.write('gllb_atomic_density_matrices',
                     pack_atomic_matrices(self.D_asp))
        writer.write('gllb_atomic_response_matrices',
                     pack_atomic_matrices(self.Dresp_asp))

        writer.write(eref_s=self.eref_s)
        writer.write(eref_source_s=self.eref_source_s)

    def empty_atomic_matrix(self):
        assert self.wfs.atom_partition is self.density.atom_partition
        return self.wfs.setups.empty_atomic_matrix(self.wfs.nspins,
                                                   self.wfs.atom_partition)

    def read(self, reader):
        r = reader.hamiltonian.xc
        wfs = self.wfs

        d('Reading vt_sG')
        self.gd.distribute(r.gllb_pseudo_response_potential / reader.ha,
                           self.vt_sG)
        self.density.distribute_and_interpolate(self.vt_sG, self.vt_sg)

        def unpack_density(D_sP):
            return unpack_atomic_matrices(D_sP, wfs.setups)

        # Read atomic density matrices and non-local part of hamiltonian:
        D_asp = unpack_density(r.gllb_atomic_density_matrices)
        Dresp_asp = unpack_density(r.gllb_atomic_response_matrices)

        # All density matrices are loaded to all cores
        # First distribute them to match density.D_asp
        self.D_asp = self.empty_atomic_matrix()
        self.Dresp_asp = self.empty_atomic_matrix()
        for a in self.density.D_asp:
            self.D_asp[a][:] = D_asp[a]
            self.Dresp_asp[a][:] = Dresp_asp[a]
        assert len(self.Dresp_asp) == len(self.density.D_asp)

        # Further distributes the density matricies for xc-corrections
        self.Ddist_asp = self.distribute_D_asp(self.D_asp)
        self.Drespdist_asp = self.distribute_D_asp(self.Dresp_asp)

        if 'eref_s' in r:
            self.eref_s = r.eref_s
            self.eref_source_s = r.eref_source_s

    def heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(self, olddens):
        # XXX This function should be removed once the deprecated
        # `fixdensity=True` option is removed.
        from gpaw.density import redistribute_array
        self.vt_sg = redistribute_array(self.vt_sg,
                                        olddens.finegd, self.finegd,
                                        self.wfs.nspins, self.wfs.kptband_comm)
        self.vt_sG = redistribute_array(self.vt_sG,
                                        olddens.gd, self.gd,
                                        self.wfs.nspins, self.wfs.kptband_comm)
