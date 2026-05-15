r"""Module for calculating electron-phonon couplings.

Electron-phonon interaction::

                  __
                  \     l   +         +
        H      =   )   g   c   c   ( a   + a  ),
         el-ph    /_    ij  i   j     l     l
                 l,ij

where the electron phonon coupling is given by::

                      ______
             l       / hbar         ___
            g   =   /-------  < i | \ /  V   * e  | j > .
             ij   \/ 2 M w           'u   eff   l
                          l

Here, l denotes the vibrational mode, w_l and e_l is the frequency and
mass-scaled polarization vector, respectively, M is an effective mass, i, j are
electronic state indices and nabla_u denotes the gradient wrt atomic
displacements. The implementation supports calculations of the el-ph coupling
in both finite and periodic systems, i.e. expressed in a basis of molecular
orbitals or Bloch states.

The implementation is based on finite-difference calculations of the the atomic
gradients of the effective potential expressed on a real-space grid. The el-ph
couplings are obtained from LCAO representations of the atomic gradients of the
effective potential and the electronic states.

In PAW the matrix elements of the derivative of the effective potential is
given by the sum of the following contributions::

                  d                  d
            < i | -- V | j > = < i | -- V | j>
                  du  eff            du

                               _
                              \        ~a     d   .       ~a
                            +  ) < i | p  >   -- /_\H   < p | j >
                              /_        i     du     ij    j
                              a,ij

                               _
                              \        d  ~a     .        ~a
                            +  ) < i | -- p  >  /_\H    < p | j >
                              /_       du  i        ij     j
                              a,ij

                               _
                              \        ~a     .        d  ~a
                            +  ) < i | p  >  /_\H    < -- p  | j >
                              /_        i        ij    du  j
                              a,ij

where the first term is the derivative of the potential (Hartree + XC) and the
last three terms originate from the PAW (pseudopotential) part of the effective
DFT Hamiltonian.

"""
from .displacements import DisplacementRunner
from .supercell import Supercell
from .gmatrix import ElectronPhononMatrix
from .raman_calculator import ResonantRamanCalculator
from .raman_data import RamanData

__all__ = [
    "DisplacementRunner",
    "Supercell",
    "ElectronPhononMatrix",
    "ResonantRamanCalculator",
    "RamanData",
]
