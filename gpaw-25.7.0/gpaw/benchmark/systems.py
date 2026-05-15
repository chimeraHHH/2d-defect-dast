import numpy as np


def system_magic_graphene():
    from gpaw.benchmark.generate_twisted import make_heterostructure
    from ase.build import graphene
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[29, -30, 0], [59, 29, 0], [0, 0, 1]])
    transb_cc = np.array([[30, -29, 0], [59, 30, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_2188_bl_graphene():
    from gpaw.benchmark.generate_twisted import make_heterostructure
    from ase.build import graphene
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[27, 13, 0], [14, 27, 0], [0, 0, 1]])
    transb_cc = np.array([[27, 24, 0], [13, 27, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_6000_bl_graphene():
    from gpaw.benchmark.generate_twisted import make_heterostructure
    from ase.build import graphene
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[23, 45, 0], [-22, 23, 0], [0, 0, 1]])
    transb_cc = np.array([[22, 45, 0], [-23, 22, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_676_bl_graphene():
    from gpaw.benchmark.generate_twisted import make_heterostructure
    from ase.build import graphene
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[7, -8, 0], [15, 7, 0], [0, 0, 1]])
    transb_cc = np.array([[8, -7, 0], [15, 8, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_H2():
    from ase.build import molecule
    atoms = molecule('H2')
    atoms.center(vacuum=3)
    return atoms


def system_C60():
    from ase.build import molecule
    atoms = molecule('C60')
    atoms.center(vacuum=5)
    return atoms


def system_diamond():
    from ase.build import bulk
    atoms = bulk('C')
    return atoms


def system_MoS2_tube():
    from math import pi
    import numpy as np
    from ase.build import mx2

    # Create tube of MoS2:
    atoms = mx2('MoS2', size=(3, 2, 1))
    atoms.cell[1, 0] = 0
    atoms = atoms.repeat((1, 10, 1))
    p = atoms.positions
    p2 = p.copy()
    L = atoms.cell[1, 1]
    r0 = L / (2 * pi)
    angle = p[:, 1] / L * 2 * pi
    p2[:, 1] = (r0 + p[:, 2]) * np.cos(angle)
    p2[:, 2] = (r0 + p[:, 2]) * np.sin(angle)
    atoms.positions = p2
    atoms.cell = [atoms.cell[0, 0], 0, 0]
    atoms.center(vacuum=6, axis=[1, 2])
    atoms.pbc = True

    return atoms


def system_magbulk():
    from ase.build import bulk
    atoms = bulk('Fe') * 2
    atoms.set_initial_magnetic_moments([3] * len(atoms))
    return atoms


def system_metalslab():
    from ase.build import fcc111
    slab = fcc111('Al', size=(3, 4, 8), vacuum=6.0)
    return slab


systems = {'C60': system_C60,
           'diamond': system_diamond,
           'H2': system_H2,
           'MoS2_tube': system_MoS2_tube,
           'C6000': system_6000_bl_graphene,
           'C2188': system_2188_bl_graphene,
           'C676': system_676_bl_graphene,
           'magbulk': system_magbulk,
           'metalslab': system_metalslab,
           'magic_graphene': system_magic_graphene}


def parse_system(name):
    return systems[name]()
