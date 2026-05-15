from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw import FermiDirac
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)


# Solvent parameters
epsinf = 78.36
gamma = 0.00114843767916
T = 298.15


def calculator():
    return SJM(
        sj={'target_potential': 4.5,
            'jelliumregion': {'top': 10.},
            'tol': 0.5},
        gpts=(8, 8, 32),
        poissonsolver={'dipolelayer': 'xy'},
        kpts=(4, 4, 1),
        xc='PBE',
        spinpol=False,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1,
                     'density': 1.0,
                     'eigenstates': 4.0,
                     'bands': 'occupied',
                     'forces': float('inf'),
                     'work function': 1},
        mode='lcao',
        basis='dzp',
        cavity=EffectivePotentialCavity(
            effective_potential=SJMPower12Potential(),
            temperature=T,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=gamma)])
