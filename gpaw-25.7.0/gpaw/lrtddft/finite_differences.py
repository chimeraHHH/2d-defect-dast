import os.path
import numpy as np
from ase import parallel as mpi

from gpaw.lrtddft.excitation import ExcitationLogger


class FiniteDifference:
    def __init__(self, atoms, propertyfunction,
                 save=False, name='fd', ending='',
                 d=0.001, parallel=1, log=None, txt='-', world=mpi.world):
        """
    atoms: Atoms object
        The atoms to work on.
    propertyfunction: function that returns a single number.
        The finite difference calculation is progressed on this value.
        For proper parallel usage the function should either be
        either a property of the atom object
            fd = FiniteDifference(atoms, atoms.property_xyz)
        or an arbitrary function with the keyword "atoms"
            fd = FiniteDifference(atoms, function_xyz)
            xyz = fd.run(atoms=atoms)
    d: float
        Magnitude of displacements.
    save: If true the write statement of the calculator is called
        to save the displacementsteps.
    name: string
        Name for restart data
    ending: string
        File handel for restart data
    parallel: int
        splits the mpi.world into 'parallel' subprocs that calculate
        displacements of different atoms individually.
    """

        self.atoms = atoms
        self.indices = np.asarray(range(len(atoms)))
        self.propertyfunction = propertyfunction
        self.save = save
        self.name = name
        self.ending = ending
        self.d = d
        self.world = world

        if log is not None:
            self.log = log
        else:
            self.log = ExcitationLogger(world=mpi.world)
            self.log.fd = txt

        if parallel > world.size:
            self.log('#', (self.__class__.__name__ + ':'),
                     'Serial calculation, keyword parallel ignored.')
            parallel = 1
        self.parallel = parallel

        assert world.size % parallel == 0

        natoms = len(self.atoms)
        self.cores_per_atom = world.size // parallel
        # my workers index
        myi = world.rank // self.cores_per_atom
        # distribute work
        self.myindices = []
        for a in range(natoms):
            if a % parallel == myi:
                self.myindices.append(a)
        # print(world.rank, 'myindices', self.myindices)

    def calculate(self, a, i, filename='fd', **kwargs):
        """Evaluate finite difference  along i'th axis on a'th atom.
        This will trigger two calls to propertyfunction(), with atom a moved
        plus/minus d in the i'th axial direction, respectively.
        if save is True the +- states are saved after
        the calculation
        """
        if 'atoms' in kwargs:
            kwargs['atoms'] = self.atoms

        p0 = self.atoms.positions[a, i]

        self.atoms.positions[a, i] += self.d
        eplus = self.propertyfunction(**kwargs)
        if self.save is True:
            savecalc = self.atoms.calc
            savecalc.write(filename + '+' + self.ending)

        self.atoms.positions[a, i] -= 2 * self.d
        eminus = self.propertyfunction(**kwargs)
        if self.save is True:
            savecalc = self.atoms.calc
            savecalc.write(filename + '-' + self.ending)
        self.atoms.positions[a, i] = p0

        self.value[a, i] = (eminus - eplus) / (2 * self.d)

        if self.parallel > 1 and self.world.rank == 0:
            self.log('# rank', mpi.world.rank, 'Atom', a,
                     'direction', i, 'FD: ', self.value[a, i])
        else:
            self.log('Atom', a, 'direction', i,
                     'FD: ', self.value[a, i])

    def run(self, **kwargs):
        """Evaluate finite differences for all atoms
        """
        self.value = np.zeros([len(self.atoms), 3])

        for filename, a, i in self.displacements():
            if a in self.myindices:
                self.calculate(a, i, filename=filename, **kwargs)

        self.world.barrier()
        self.value /= self.cores_per_atom
        self.world.sum(self.value)

        return self.value

    def displacements(self):
        for a in self.indices:
            for i in range(3):
                filename = ('{}_{}_{}'.format(self.name, a, 'xyz'[i]))
                yield filename, a, i

    def restart(self, restartfunction, **kwargs):
        """Uses restartfunction to recalculate values
        from the saved files.
        If a file with the corresponding name is found the
        restartfunction is called to get the FD value
        The restartfunction should take a string as input
        parameter like the standart read() function.
        If no file is found, a calculation is initiated.
        Example:
            def re(self, name):
                calc = Calculator(restart=name)
                return calc.get_potential_energy()

            fd = FiniteDifference(atoms, atoms.get_potential_energy)
            fd.restart(re)
        """
        for filename, a, i in self.displacements():

            if (os.path.isfile(filename + '+' + self.ending) and
                    os.path.isfile(filename + '-' + self.ending)):
                eplus = restartfunction(
                    self, filename + '+' + self.ending, **kwargs)
                eminus = restartfunction(
                    self, filename + '-' + self.ending, **kwargs)
                self.value[a, i] = (eminus - eplus) / (2 * self.d)
            else:
                self.calculate(a, i, filename=filename, **kwargs)

        return self.value
