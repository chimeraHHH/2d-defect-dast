import pytest


def test_gap_in_txt(gpw_files, needs_ase_master):
    gpw = gpw_files['h2_pw']
    txt = gpw.with_suffix('.txt')
    for line in txt.read_text().splitlines():
        if line.startswith('Gap:'):
            gap = float(line.split()[1])
            assert gap == pytest.approx(11.296, abs=0.001)
            return
    raise ValueError(f'No gap in text file: {txt}')
