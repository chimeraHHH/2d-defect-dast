import numpy as np
from ase.units import Ha
from gpaw.xc.functional import XCFunctional


class NonLocalFunctional(XCFunctional):
    def __init__(self, xcname, setup_name=None):
        self.contributions = []
        # TODO: remove self.xcs once deprecated calculate_delta_xc()
        # in c_response.py is removed
        self.xcs = {}
        XCFunctional.__init__(self, xcname, 'GLLB')
        if setup_name is None:
            self.setup_name = self.name
        else:
            self.setup_name = setup_name
        self.mix = None
        self.mix_vt_sg = None
        self.old_vt_sg = None
        self.old_H_asp = {}
        self.response = None

    def get_setup_name(self):
        return self.setup_name

    def set_mix(self, mix):
        self.mix = mix

    def initialize(self, density, hamiltonian, wfs):
        for contribution in self.contributions:
            contribution.initialize(density, hamiltonian, wfs)

    def initialize_1d(self, ae):
        for contribution in self.contributions:
            contribution.initialize_1d(ae)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        e_g[:] = 0.0
        if self.mix is None:
            for contribution in self.contributions:
                contribution.calculate(e_g, n_sg, v_sg)
        else:
            cmix = self.mix
            if self.mix_vt_sg is None:
                self.mix_vt_sg = np.zeros_like(v_sg)
                self.old_vt_sg = np.zeros_like(v_sg)
                cmix = 1.0
            self.mix_vt_sg[:] = 0.0
            for contribution in self.contributions:
                contribution.calculate(e_g, n_sg, self.mix_vt_sg)
            self.mix_vt_sg = (cmix * self.mix_vt_sg
                              + (1.0 - cmix) * self.old_vt_sg)
            v_sg += self.mix_vt_sg
            self.old_vt_sg[:] = self.mix_vt_sg

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp, a=None,
                                 addcoredensity=True):
        return self.calculate_energy_and_derivatives(setup, D_sp, dEdD_sp, a,
                                                     addcoredensity)

    def calculate_energy_and_derivatives(self, setup, D_sp, H_sp, a,
                                         addcoredensity=True):
        if setup.xc_correction is None:
            return 0.0
        Exc = 0.0
        # We are supposed to add to H_sp, not write directly
        H0_sp = H_sp
        H_sp = H0_sp.copy()
        H_sp[:] = 0.0

        if self.mix is None:
            for contribution in self.contributions:
                Exc += contribution.calculate_energy_and_derivatives(
                    setup, D_sp, H_sp, a, addcoredensity)
        else:
            cmix = self.mix
            if a not in self.old_H_asp:
                self.old_H_asp[a] = H_sp.copy()
                cmix = 1.0

            for contribution in self.contributions:
                Exc += contribution.calculate_energy_and_derivatives(
                    setup, D_sp, H_sp, a, addcoredensity)
            H_sp *= cmix
            H_sp += (1 - cmix) * self.old_H_asp[a]
            self.old_H_asp[a][:] = H_sp.copy()

        H0_sp += H_sp
        Exc -= setup.xc_correction.e_xc0
        return Exc

    def get_xc_potential_and_energy_1d(self, v_g):
        Exc = 0.0
        for contribution in self.contributions:
            Exc += contribution.add_xc_potential_and_energy_1d(v_g)
        return Exc

    def get_smooth_xc_potential_and_energy_1d(self, vt_g):
        Exc = 0.0
        for contribution in self.contributions:
            Exc += contribution.add_smooth_xc_potential_and_energy_1d(vt_g)
        return Exc

    def initialize_from_atomic_orbitals(self, basis_functions):
        for contribution in self.contributions:
            contribution.initialize_from_atomic_orbitals(basis_functions)

    def get_extra_setup_data(self, extra_data):
        for contribution in self.contributions:
            contribution.get_extra_setup_data(extra_data)

    def add_contribution(self, contribution):
        self.contributions.append(contribution)
        self.xcs[contribution.get_name()] = contribution
        if contribution.get_name() == 'RESPONSE':
            assert self.response is None
            self.response = contribution

    def get_description(self):
        fmt = '| %-6s | %-10s | %-45s |'
        header = fmt % ('Weight', 'Module', 'Description')
        dashes = '-' * len(header)
        s = ['{} functional being used consists of'.format(self.name)]
        s += [dashes]
        s += [header]
        s += [dashes]
        for c in self.contributions:
            s += [fmt % ('%6.3f' % c.weight, c.get_name(), c.get_desc())]
        s += [dashes]
        return '\n'.join(s)

    def summary(self, log):
        if self.response is not None:
            eref_s = self.response.eref_s
            source_s = self.response.eref_source_s
            if eref_s is not None and source_s is not None:
                text_s = [f'{eref * Ha:.5f} eV (from {source})'
                          for eref, source in zip(eref_s, source_s)]
                log('{} response reference energy: {}\n'
                    .format(self.name, ', '.join(text_s)))

    def read(self, reader):
        for contribution in self.contributions:
            contribution.read(reader)

    def write(self, writer):
        for contribution in self.contributions:
            contribution.write(writer)

    def heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(self, olddens):
        # XXX This function should be removed once the deprecated
        # `fixdensity=True` option is removed.
        for contribution in self.contributions:
            try:
                contribution.heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(olddens)
            except AttributeError:
                pass
