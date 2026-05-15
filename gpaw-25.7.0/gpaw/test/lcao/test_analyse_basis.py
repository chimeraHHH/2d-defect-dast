from pathlib import Path

import pytest

from gpaw.lcao.analyse_basis import main


def test_analyse_basis(in_tmp_dir, capsys):
    pytest.importorskip('matplotlib')
    argv = ['H.dzp.basis', '--save-figs']
    main(argv)
    captured = capsys.readouterr()
    out = captured.out
    assert '1s-sz' in out
    assert '1s-dz' in out
    assert 'p-type' in out
    assert Path('H.dzp.png').is_file()
