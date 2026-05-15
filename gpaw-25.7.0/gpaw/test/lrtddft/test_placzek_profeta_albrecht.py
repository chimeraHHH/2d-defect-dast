import pytest
from ase import Atom, Atoms

from gpaw import GPAW
from gpaw.analyse.overlap import Overlap
from gpaw.lrtddft.kssingle import KSSingles

txt = '-'
txt = None
load = True
load = False
xc = 'LDA'

# run

R = 0.7  # approx. experimental bond length
a = 4.0
c = 5.0


@pytest.mark.lrtddft
def test_lrtddft_placzek_profeta_albrecht(in_tmp_dir):
    from ase.vibrations.albrecht import Albrecht
    from ase.vibrations.placzek import Placzek, Profeta
    from ase.vibrations.resonant_raman import ResonantRamanCalculator

    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))

    name = exname = 'rraman'
    exkwargs = {'restrict': {'eps': 0.0, 'jend': 3}}

    calc = GPAW(
        mode='fd',
        xc=xc, nbands=7,
        convergence={'bands': 3},
        spinpol=False,
        # eigensolver='rmm-diis',
        symmetry={'point_group': False},
        txt=txt)
    H2.calc = calc
    # H2.get_potential_energy()

    rr = ResonantRamanCalculator(
        H2, KSSingles, name=name, exname=exname,
        exkwargs=exkwargs,
        # XXX full does not work in parallel due to boxes
        # on different nodes
        # overlap=lambda x, y: Overlap(x).full(y)[0],
        overlap=lambda x, y: Overlap(x).pseudo(y)[0],
        txt=txt)
    rr.run()

    # check

    # Different Placzeck implementations should agree

    om = 5
    pz = Placzek(H2, KSSingles, name=name, exname=exname, txt=txt)
    pzi = pz.get_absolute_intensities(omega=om)[-1]

    pr = Profeta(H2, KSSingles, name=name, exname=exname,
                 approximation='Placzek', txt=txt)
    pri = pr.get_absolute_intensities(omega=om)[-1]
    assert pzi == pytest.approx(pri, abs=0.1)

    pr = Profeta(H2, KSSingles, name=name, exname=exname,
                 overlap=True,
                 approximation='Placzek', txt=txt)
    pri = pr.get_absolute_intensities(omega=om)[-1]
    assert pzi == pytest.approx(pri, abs=0.1)

    """Albrecht and Placzek are approximately equal"""

    al = Albrecht(H2, KSSingles, name=name, exname=exname,
                  overlap=True,
                  approximation='Albrecht', txt=txt)
    ali = al.get_absolute_intensities(omega=om)[-1]
    assert pzi == pytest.approx(ali, abs=1.5)

    """Albrecht A and P-P are approximately equal"""

    pr = Profeta(H2, KSSingles, name=name, exname=exname,
                 overlap=True,
                 approximation='P-P', txt=txt)
    pri = pr.get_absolute_intensities(omega=om)[-1]

    al = Albrecht(H2, KSSingles, name=name, exname=exname,
                  overlap=True,
                  approximation='Albrecht A', txt=txt)
    ali = al.get_absolute_intensities(omega=om)[-1]
    assert pri == pytest.approx(ali, abs=3)

    """Albrecht B+C and Profeta are approximately equal"""

    pr = Profeta(H2, KSSingles, name=name, exname=exname,
                 overlap=True,
                 approximation='Profeta', txt=txt)
    pri = pr.get_absolute_intensities(omega=om)[-1]

    al = Albrecht(H2, KSSingles, name=name, exname=exname,
                  overlap=True,
                  approximation='Albrecht BC', txt=txt)
    ali = al.get_absolute_intensities(omega=om)[-1]
    assert pri == pytest.approx(ali, abs=3)
