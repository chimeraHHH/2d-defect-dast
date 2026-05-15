import pytest
import numpy as np
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.dipolemomentwriter import VelocityGaugeWriter


@pytest.mark.rttddft
@pytest.mark.parametrize('propagator', ['sicn', 'scpc', 'ecn'])
def test_propagators(propagator, gpw_files, in_tmp_dir):
    # XXX convergence={'density': 1e-8} originally 1e-12
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'], propagator=propagator)
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(40, 20)
    data = np.loadtxt('dm.dat')

    # Make sure that norm and x and y components are zero
    assert data[:, 1:4] == pytest.approx(0, abs=1e-8)
    # Isolate z-directional data for comparison
    data_i = data[:, 4]

    if propagator == 'sicn':
        ref_i = [1.440334447474e-15,
                 1.125313930892e-15,
                 3.584347565411e-05,
                 7.109522259009e-05,
                 1.051768025451e-04,
                 1.375320463618e-04,
                 1.676368375148e-04,
                 1.950075908129e-04,
                 2.192083357657e-04,
                 2.398564722712e-04,
                 2.566273642857e-04,
                 2.692579717986e-04,
                 2.775497108433e-04,
                 2.813706965647e-04,
                 2.806574778736e-04,
                 2.754163145186e-04,
                 2.657239704053e-04,
                 2.517278993102e-04,
                 2.336455958733e-04,
                 2.117628070554e-04,
                 1.864302841758e-04,
                 1.580588287559e-04]

    elif propagator == 'scpc':
        ref_i = [1.440334447474e-15,
                 1.125313930892e-15,
                 3.584439590353e-05,
                 7.109886048577e-05,
                 1.051848309578e-04,
                 1.375459452388e-04,
                 1.676578426631e-04,
                 1.950366598449e-04,
                 2.192461280005e-04,
                 2.399033382662e-04,
                 2.566833411562e-04,
                 2.693227827882e-04,
                 2.776227689310e-04,
                 2.814510134936e-04,
                 2.807437787678e-04,
                 2.755072543407e-04,
                 2.658179386847e-04,
                 2.518229288423e-04,
                 2.337394843530e-04,
                 2.118531419552e-04,
                 1.865144815236e-04,
                 1.581341901411e-04]

    elif propagator == 'ecn':
        ref_i = [1.440334447474e-15,
                 1.125313930892e-15,
                 3.592248075296e-05,
                 7.140807909782e-05,
                 1.058673077088e-04,
                 1.387243364676e-04,
                 1.694288472618e-04,
                 1.974671596663e-04,
                 2.223720449505e-04,
                 2.437308555076e-04,
                 2.611918303868e-04,
                 2.744686373485e-04,
                 2.833433443564e-04,
                 2.876680791149e-04,
                 2.873656575748e-04,
                 2.824294614402e-04,
                 2.729228248343e-04,
                 2.589781389459e-04,
                 2.407957786298e-04,
                 2.186427967116e-04,
                 1.928511441932e-04,
                 1.638150141478e-04]

    assert data_i == pytest.approx(ref_i, abs=1e-8)


@pytest.mark.serial  # Todo:remove later
@pytest.mark.rttddft
def test_velocity(gpw_files, in_tmp_dir):

    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'])

    VelocityGaugeWriter(td_calc, 'dm_velocityGauge.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5], gauge='velocity')
    td_calc.propagate(10, 20)
    data_i = np.loadtxt('dm_velocityGauge.dat')[:, 4]
    ref_i = [0.000000000000e+00,
             0.000000000000e+00,
             1.006067165210e-08,
             4.016494936588e-08,
             9.010441886809e-08,
             1.595392493537e-07,
             2.480091681830e-07,
             3.549482463863e-07,
             4.797029544063e-07,
             6.215528195238e-07,
             7.797329169342e-07,
             9.534573665756e-07,
             1.141942968879e-06,
             1.344432105004e-06,
             1.560214064258e-06,
             1.788644005768e-06,
             2.029158870610e-06,
             2.281289660947e-06,
             2.544669639498e-06,
             2.819038164914e-06,
             3.104240038438e-06,
             3.400220392727e-06]
    assert data_i == pytest.approx(ref_i, abs=1e-10)
    from gpaw.tddft.spectrum import photoabsorption_spectrum
    photoabsorption_spectrum('dm_velocityGauge.dat', 'spec.dat')
    data_i = np.loadtxt('spec.dat')
    assert pytest.approx(-7.68e-7, 1e-5, 1e-9) == data_i[1, 3]
    assert pytest.approx(-3.07e-6, 1e-5, 1e-8) == data_i[2, 3]
