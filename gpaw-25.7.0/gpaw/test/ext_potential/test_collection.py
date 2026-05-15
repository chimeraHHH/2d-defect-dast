from gpaw.grid_descriptor import GridDescriptor
from gpaw.external import (ConstantElectricField, PointChargePotential,
                           PotentialCollection, create_external_potential)


def test_collection():
    a = 4.0
    N = 48
    gd = GridDescriptor((N, N, N), (a, a, a), 0)

    ext1 = ConstantElectricField(1)
    ext2 = PointChargePotential([1, -5], positions=((0, 0, -10), (0, 0, 10)))
    collection = PotentialCollection([ext1, ext2])

    ext1.calculate_potential(gd)
    ext2.calculate_potential(gd)
    collection.calculate_potential(gd)

    assert (collection.vext_g == ext1.vext_g + ext2.vext_g).all()

    assert len(collection.todict()['potentials']) == 2
    for text in ['Constant electric', 'Point-charge',
                 collection.__class__.__name__]:
        assert text in collection.__str__()

    # reconstruct
    collection2 = create_external_potential(**collection.todict())
    collection2.calculate_potential(gd)
    assert (collection2.vext_g == collection.vext_g).all()
