import numpy as np
from gpaw.core import PWDesc
from gpaw.mpi import broadcast_float
from gpaw.new import zips, spinsum, trace
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.stress import calculate_stress
from gpaw.setup import Setups


class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 grid,
                 fine_grid,
                 pw: PWDesc,
                 setups: Setups,
                 xc,
                 poisson_solver,
                 *,
                 relpos_ac,
                 atomdist,
                 environment,
                 extensions,
                 soc=False,
                 xp=np):
        self.xp = xp
        self.pw = pw
        super().__init__(xc, poisson_solver, setups,
                         relpos_ac=relpos_ac,
                         environment=environment,
                         extensions=extensions,
                         soc=soc)

        self.vbar_ag = setups.create_local_potentials(
            pw, relpos_ac, atomdist, xp)

        self.fftplan = grid.fft_plans(xp=xp)
        self.fftplan2 = fine_grid.fft_plans(xp=xp)

        self.grid = grid
        self.fine_grid = fine_grid

        self.vbar_g = pw.zeros(xp=xp)
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()

        # For forces and stress:
        self._nt_g = None
        self._vt_g = None
        self._dedtaut_g = None

    def interpolate(self, a_R, a_r=None):
        return a_R.interpolate(self.fftplan, self.fftplan2,
                               grid=self.fine_grid, out=a_r)

    def restrict(self, a_r, a_R=None):
        return a_r.fft_restrict(self.fftplan2, self.fftplan,
                                grid=self.grid, out=a_R)

    def _interpolate_density(self, nt_sR):
        nt_sr = self.fine_grid.empty(nt_sR.dims, xp=self.xp)
        pw = self.vbar_g.desc

        if pw.comm.rank == 0:
            pw0 = self.poisson_solver.pwg0
            indices = self.xp.asarray(pw0.indices(self.fftplan.shape))
            nt0_g = pw0.zeros(xp=self.xp)
        else:
            nt0_g = None

        ndensities = nt_sR.dims[0] % 3
        for spin, (nt_R, nt_r) in enumerate(zips(nt_sR, nt_sr)):
            self.interpolate(nt_R, nt_r)
            if spin < ndensities and pw.comm.rank == 0:
                nt0_g.data += self.xp.asarray(
                    self.fftplan.tmp_Q.ravel()[indices])

        return nt_sr, nt0_g

    def _interpolate_and_calculate_xc(self, xc, density):
        nt_sr, nt0_g = self._interpolate_density(density.nt_sR)

        if density.taut_sR is not None:
            taut_sr = self.interpolate(density.taut_sR)
        else:
            taut_sr = None

        e_xc, vxct_sr, dedtaut_sr = xc.calculate(nt_sr, taut_sr)

        return nt_sr, nt0_g, taut_sr, e_xc, vxct_sr, dedtaut_sr

    def calculate_non_selfconsistent_exc(self, xc, density):
        _, _, _, e_xc, _, _ = self._interpolate_and_calculate_xc(xc, density)
        return e_xc

    @trace
    def calculate_pseudo_potential(self, density, ibzwfs, vHt_h=None):
        nt_sr, nt0_g, taut_sr, e_xc, vxct_sr, dedtaut_sr = (
            self._interpolate_and_calculate_xc(self.xc, density))

        pw = self.vbar_g.desc
        if pw.comm.rank == 0:
            nt0_g.data *= 1 / np.prod(density.nt_sR.desc.size_c)
            e_zero = self.vbar0_g.integrate(nt0_g)
        else:
            e_zero = 0.0
        e_zero = broadcast_float(float(e_zero), pw.comm)

        if pw.comm.rank == 0:
            vt0_g = self.vbar0_g.copy()
        else:
            vt0_g = None

        self.environment.update1pw(nt0_g)

        Q_aL = density.calculate_compensation_charge_coefficients()

        e_coulomb, vHt_h, V_aL = self.poisson_solver.solve(
            nt0_g, Q_aL, vt0_g, vHt_h)

        if pw.comm.rank == 0:
            vt0_R = vt0_g.ifft(
                plan=self.fftplan,
                grid=density.nt_sR.desc.new(comm=None))

        vt_sR = density.nt_sR.new()
        vt_sR[0].scatter_from(vt0_R if pw.comm.rank == 0 else None)
        if density.ndensities == 2:
            vt_sR.data[1] = vt_sR.data[0]
        vt_sR.data[density.ndensities:] = 0.0

        # e_external = self.external_potential.update_potential(vt_sR, density)
        e_external = 0.0

        vtmp_R = vt_sR.desc.empty(xp=self.xp)
        for spin, (vt_R, vxct_r) in enumerate(zips(vt_sR, vxct_sr)):
            self.restrict(vxct_r, vtmp_R)
            vt_R.data += vtmp_R.data

        self._reset()

        e_stress = e_coulomb + e_zero

        return ({'coulomb': e_coulomb,
                 'zero': e_zero,
                 'xc': e_xc,
                 'external': e_external},
                vt_sR,
                dedtaut_sr,
                vHt_h,
                V_aL,
                e_stress)

    def move(self, relpos_ac, atomdist):
        super().move(relpos_ac, atomdist)
        self.poisson_solver.move(relpos_ac, atomdist)
        self.vbar_ag.move(relpos_ac, atomdist)
        self.vbar_g.data[:] = 0.0
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()
        self._reset()

    def _reset(self):
        self._vt_g = None
        self._nt_g = None
        self._dedtaut_g = None

    def _force_stress_helper(self, density, potential):
        # Only do the work once - in case both forces and stresses are needed:
        if self._vt_g is not None:
            return self._vt_g, self._nt_g, self._dedtaut_g

        nt_R = spinsum(density.nt_sR)
        vt_R = spinsum(potential.vt_sR, mean=True)
        self._vt_g = vt_R.fft(self.fftplan, pw=self.pw)
        self._nt_g = nt_R.fft(self.fftplan, pw=self.pw)

        dedtaut_sR = potential.dedtaut_sR
        if dedtaut_sR is not None:
            dedtaut_R = spinsum(dedtaut_sR, mean=True)
            self._dedtaut_g = dedtaut_R.fft(self.fftplan, pw=self.pw)
        else:
            self._dedtaut_g = None

        return self._vt_g, self._nt_g, self._dedtaut_g

    def force_contributions(self, Q_aL, density, potential):
        if potential.vHt_x is None:
            raise RuntimeError(ERROR.format(thing='forces'))
        vt_g, nt_g, dedtaut_g = self._force_stress_helper(density, potential)
        if dedtaut_g is None:
            Ftauct_av = None
        else:
            Ftauct_av = density.tauct_aX.derivative(dedtaut_g)

        return (
            self.poisson_solver.force_contribution(Q_aL,
                                                   potential.vHt_x,
                                                   nt_g),
            density.nct_aX.derivative(vt_g),
            Ftauct_av,
            self.vbar_ag.derivative(nt_g),
            self.extensions_force_av)

    def stress(self, ibzwfs, density, potential):
        if potential.vHt_x is None:
            raise RuntimeError(ERROR.format(thing='stress'))
        vt_g, nt_g, dedtaut_g = self._force_stress_helper(density, potential)
        return calculate_stress(self, ibzwfs, density, potential,
                                vt_g, nt_g, dedtaut_g)


ERROR = (
    'Unable to calculate {thing}.  Are you restartting from an old '
    'gpw-file?  In that case, calculate the {thing} before writing '
    'the gpw-file or switch to new GPAW.')
