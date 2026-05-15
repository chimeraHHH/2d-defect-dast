from ase.data import chemical_symbols
from ase.units import Bohr, Hartree
from gpaw import GPAW_NEW
from gpaw.calculator import GPAW as OldGPAW
from gpaw.io import Reader
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian


def SolvationGPAW(*args, **kwargs):
    if GPAW_NEW:
        from gpaw.new.ase_interface import GPAW
        solvation = dict(name='solvation',
                         cavity=kwargs.pop('cavity'),
                         dielectric=kwargs.pop('dielectric'),
                         interactions=kwargs.pop('interactions', None))
        return GPAW(*args, **kwargs, environment=solvation)
    return OldSolvationGPAW(*args, **kwargs)


class OldSolvationGPAW(OldGPAW):
    """Subclass of gpaw.GPAW calculator with continuum solvent model.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    def __init__(self, restart=None, cavity=None, dielectric=None,
                 interactions=None, **gpaw_kwargs):
        """Constructor for SolvationGPAW class.

        Additional arguments not present in GPAW class:
        cavity       -- A Cavity instance.
        dielectric   -- A Dielectric instance.
        interactions -- A list of Interaction instances.
        """
        if interactions is None:
            interactions = []

        # if not all([cavity, dielectric]):
        #    raise IOError('Cavity and dielectric modules need to be '
        #                  'defined in the calculator')

        self.stuff_for_hamiltonian = (cavity, dielectric, interactions)

        OldGPAW.__init__(self, restart, **gpaw_kwargs)

        self.log('Implicit solvation parameters:')
        for stuff in self.stuff_for_hamiltonian:
            if isinstance(stuff, list):
                for instuff in stuff:
                    self.log(instuff)
            else:
                self.log(stuff)
        self.log()

    def read(self, filename):
        """Read yourself from a file"""
        self.reader = reader = Reader(filename)
        if 'implicit_solvent' in reader:
            impl_in = reader.implicit_solvent
            if 'name' in impl_in.cavity.effective_potential:
                efpot = impl_in.cavity.effective_potential

                atomic_radii = {}
                for Z, r in zip(reader.atoms.numbers, efpot.atomic_radii):
                    symbol = chemical_symbols[Z]
                    if symbol in atomic_radii:
                        assert atomic_radii[symbol] == r
                    else:
                        atomic_radii[symbol] = r

                if efpot.name == 'SJMPower12Potential':
                    from gpaw.solvation.sjm import SJMPower12Potential

                    effective_potential = SJMPower12Potential(
                        atomic_radii=atomic_radii,
                        u0=efpot.u0,
                        H2O_layer=efpot.H2O_layer,
                        unsolv_backside=efpot.unsolv_backside)
                elif efpot.name == 'Power12Potential':
                    from gpaw.solvation.sjm import Power12Potential
                    effective_potential = Power12Potential(
                        atomic_radii=atomic_radii,
                        u0=efpot.u0)
                else:
                    raise OSError('Reading the given effective potential '
                                  'is not implemented yet')

            if 'name' in impl_in.cavity.surface_calculator:
                suca = impl_in.cavity.surface_calculator
                if suca.name == 'GradientSurface':
                    from gpaw.solvation.cavity import GradientSurface
                    surface_calculator = GradientSurface(suca.nn)
                else:
                    raise OSError('Reading in the given used surface '
                                  'calculator is not implemented')

            T = impl_in.cavity.temperature

            from gpaw.solvation.cavity import EffectivePotentialCavity
            cavity = EffectivePotentialCavity(
                effective_potential=effective_potential,
                temperature=T,
                surface_calculator=surface_calculator)

            if impl_in.dielectric.name == 'LinearDielectric':
                from gpaw.solvation.dielectric import LinearDielectric
                dielectric = LinearDielectric(epsinf=impl_in.dielectric.epsinf)

            if impl_in.interactions.name == 'SurfaceInteraction':
                suin = impl_in.interactions
                from gpaw.solvation.interactions import SurfaceInteraction
                interactions = [SurfaceInteraction(suin.surface_tension)]

            self.stuff_for_hamiltonian = (cavity, dielectric, interactions)

        reader = OldGPAW.read(self, filename)
        return reader

    def _write(self, writer, mode):
        OldGPAW._write(self, writer, mode)
        stuff = self.stuff_for_hamiltonian
        writer.child('implicit_solvent').write(cavity=stuff[0],
                                               dielectric=stuff[1],
                                               interactions=stuff[2][0])

    def create_hamiltonian(self, realspace, mode, xc):
        if not realspace:
            raise NotImplementedError(
                'SolvationGPAW does not support '
                'calculations in reciprocal space yet.')

        dens = self.density
        self.hamiltonian = SolvationRealSpaceHamiltonian(
            *self.stuff_for_hamiltonian,
            gd=dens.gd, finegd=dens.finegd,
            nspins=dens.nspins,
            collinear=dens.collinear,
            setups=dens.setups,
            timer=self.timer,
            xc=xc,
            world=self.world,
            redistributor=dens.redistributor,
            vext=self.parameters.external,
            psolver=self.parameters.poissonsolver,
            stencil=mode.interpolation)

        self.log(self.hamiltonian)

        xc.set_grid_descriptor(self.hamiltonian.finegd)

    def initialize_positions(self, atoms=None):
        spos_ac = OldGPAW.initialize_positions(self, atoms)
        self.hamiltonian.update_atoms(self.atoms, self.log)
        return spos_ac

    def get_electrostatic_energy(self):
        """Return electrostatic part of the total energy.

        The electrostatic part consists of everything except
        the short-range interactions defined in the interactions list.
        """
        # Energy extrapolated to zero width:
        return Hartree * self.hamiltonian.e_el_extrapolated

    def get_solvation_interaction_energy(self, subscript, atoms=None):
        """Return a specific part of the solvation interaction energy.

        The subscript parameter defines which part is to be returned.
        It has to match the value of a subscript attribute of one of
        the interactions in the interactions list.
        """
        return Hartree * getattr(self.hamiltonian, 'e_' + subscript)

    def get_cavity_volume(self, atoms=None):
        """Return the cavity volume in Angstrom ** 3.

        In case no volume calculator has been set for the cavity, None
        is returned.
        """
        V = self.hamiltonian.cavity.V
        return V and V * Bohr ** 3

    def get_cavity_surface(self, atoms=None):
        """Return the cavity surface area in Angstrom ** 2.

        In case no surface calculator has been set for the cavity,
        None is returned.
        """
        A = self.hamiltonian.cavity.A
        return A and A * Bohr ** 2
