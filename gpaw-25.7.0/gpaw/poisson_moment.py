from typing import Any, Dict, Optional, List, Sequence, Union

import numpy as np
from ase.units import Bohr
from ase.utils.timing import Timer
from gpaw.poisson import _PoissonSolver, create_poisson_solver
from gpaw.utilities.gauss import Gaussian
from gpaw.typing import Array1D
from gpaw.utilities.timing import nulltimer, NullTimer

from ase.utils.timing import timer


MomentCorrectionsType = Union[int, List[Dict[str, Any]]]


class MomentCorrection:

    def __init__(self,
                 center: Optional[Union[Sequence, Array1D]],
                 moms: Union[int, Sequence[int]]):
        if center is not None:
            center = np.asarray(center) / Bohr
        self.center = center
        self.moms = np.asarray(moms)

    def todict(self) -> Dict[str, Any]:
        """ return dictionary description, converting the moment correction
        from units of Bohr to Ångström """

        center = self.center
        if center is not None:
            center = center * Bohr

        dict_out = dict(moms=self.moms, center=center)

        return dict_out

    def __str__(self) -> str:
        if self.center is None:
            center = 'center'
        else:
            center = ', '.join([f'{x:.2f}' for x in self.center * Bohr])

        if np.allclose(np.diff(self.moms), 1):
            # Increasing sequence
            moms = f'range({self.moms[0]}, {self.moms[-1] + 1})'
        else:
            # List of integers
            _moms = ', '.join([f'{m:.0f}' for m in self.moms])
            moms = f'({_moms})'
        return f'[{center}] {moms}'

    def __repr__(self) -> str:
        center = self.center
        if center is not None:
            center *= Bohr
        return f'{repr(self.moms)} @ {repr(center)}'


class MomentCorrectionPoissonSolver(_PoissonSolver):
    """Wrapper for the poisson solver that includes moment corrections

    Parameters
    ----------
    poissonsolver
        underlying poisson solver
    moment_corrections
        list of moment corrections, expressed as dictionaries
        `{'moms': ..., 'center': ...}` that specify the multipole moments
        and their centres.

        moment_corrections = [{'moms': moms_list1, 'center': center1},
                              {'moms': moms_list2, 'center': center2},
                              ...]

        Here moms_listX is list of integers of multipole moments to be
        corrected at centerX. For example moms_list=range(4) corresponds to
        s, p_x, p_y and p_z type multipoles.

        Optionally setting moment_corrections to an integer m is equivalent to
        including multipole moments corresponding to range(m) at the center of
        the cell


    timer
        timer

    """

    def __init__(self,
                 poissonsolver: Union[_PoissonSolver, Dict[str, Any]],
                 moment_corrections: Optional[MomentCorrectionsType],
                 timer: Union[NullTimer, Timer] = nulltimer):

        self._initialized = False
        self.poissonsolver = create_poisson_solver(poissonsolver)
        self.timer = timer

        if moment_corrections is None:
            self.moment_corrections = []
        elif isinstance(moment_corrections, int):
            moms = range(moment_corrections)
            center = None
            self.moment_corrections = [MomentCorrection(moms=moms,
                                                        center=center)]
        elif isinstance(moment_corrections, list):
            assert all(['moms' in mom and 'center' in mom
                        for mom in moment_corrections]), \
                   (f'{self.__class__.__name__}: each element in '
                    'moment_correction must be a dictionary '
                    'with the keys "moms" and "center"')
            self.moment_corrections = [MomentCorrection(**mom)
                                       for mom in moment_corrections]
        else:
            raise ValueError(f'{self.__class__.__name__}: moment_correction '
                             'must be a list of dictionaries')

    def todict(self):
        mom_corr = [mom.todict() for mom in self.moment_corrections]
        d = {'name': 'MomentCorrectionPoissonSolver',
             'poissonsolver': self.poissonsolver.todict(),
             'moment_corrections': mom_corr}

        return d

    def set_grid_descriptor(self, gd):
        self.poissonsolver.set_grid_descriptor(gd)
        self.gd = gd

    def get_description(self) -> str:
        description = self.poissonsolver.get_description()
        n = len(self.moment_corrections)

        lines = [description]
        lines.extend([f'    {n} moment corrections:'])
        lines.extend([f'      {str(mom)}'
                     for mom in self.moment_corrections])

        return '\n'.join(lines)

    @timer('Poisson initialize')
    def _init(self):
        if self._initialized:
            return
        self.poissonsolver._init()

        if not self.gd.orthogonal or self.gd.pbc_c.any():
            raise NotImplementedError('Only orthogonal unit cells '
                                      'and non-periodic boundary '
                                      'conditions are tested')
        self.load_moment_corrections_gauss()

        self._initialized = True

    @timer('Load moment corrections')
    def load_moment_corrections_gauss(self):
        self.gauss_i = []
        self.mom_ij = []
        self.mask_ig = []

        if len(self.moment_corrections) == 0:
            return

        mask_ir = []
        r_ir = []

        for rmom in self.moment_corrections:
            center = rmom.center
            mom_j = rmom.moms
            gauss = Gaussian(self.gd, center=center)
            self.gauss_i.append(gauss)
            r_ir.append(gauss.r.ravel())
            mask_ir.append(self.gd.zeros(dtype=int).ravel())
            self.mom_ij.append(mom_j)

        r_ir = np.array(r_ir)
        mask_ir = np.array(mask_ir)

        Ni = r_ir.shape[0]
        Nr = r_ir.shape[1]

        for r in range(Nr):
            i = np.argmin(r_ir[:, r])
            mask_ir[i, r] = 1

        for i in range(Ni):
            mask_r = mask_ir[i]
            mask_g = mask_r.reshape(self.gd.n_c)
            self.mask_ig.append(mask_g)

    def solve(self, phi, rho, **kwargs):
        self._init()
        return self._solve(phi, rho, **kwargs)

    @timer('Solve')
    def _solve(self, phi, rho, **kwargs):
        timer = kwargs.get('timer', self.timer)

        if len(self.moment_corrections) > 0:
            assert not self.gd.pbc_c.any()

            timer.start('Multipole moment corrections')

            rho_neutral = rho * 0.0
            phi_cor_g = self.gd.zeros()
            for gauss, mask_g, mom_j in zip(self.gauss_i, self.mask_ig,
                                            self.mom_ij):
                rho_masked = rho * mask_g
                for mom in mom_j:
                    phi_cor_g += gauss.remove_moment(rho_masked, mom)
                rho_neutral += rho_masked

            # Remove multipoles for better initial guess
            phi -= phi_cor_g

            timer.stop('Multipole moment corrections')

            timer.start('Solve neutral')
            niter = self.poissonsolver.solve(phi, rho_neutral, **kwargs)
            timer.stop('Solve neutral')

            timer.start('Multipole moment corrections')
            # correct error introduced by removing multipoles
            phi += phi_cor_g
            timer.stop('Multipole moment corrections')

            return niter
        else:
            return self.poissonsolver.solve(phi, rho, **kwargs)

    def estimate_memory(self, mem):
        self.poissonsolver.estimate_memory(mem)
        gdbytes = self.gd.bytecount()
        if self.moment_corrections is not None:
            mem.subnode('moment_corrections masks',
                        len(self.moment_corrections) * gdbytes)

    def __repr__(self):
        if len(self.moment_corrections) == 0:
            corrections_str = 'no corrections'
        elif len(self.moment_corrections) < 2:
            corrections_str = f'{repr(self.moment_corrections[0])}'
        else:
            corrections_str = f'{len(self.moment_corrections)} corrections'

        representation = f'MomentCorrectionPoissonSolver ({corrections_str})'
        return representation
