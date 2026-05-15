import subprocess

import pytest

from gpaw.basis_data import parse_basis_filename
from gpaw.setup_data import search_for_file


def test_parse_basis_filename():
    assert parse_basis_filename('Si.dzp.basis') == ('Si', 'dzp')
    assert parse_basis_filename('Si.basis') == ('Si', None)
    assert parse_basis_filename('Si.any.thing.basis') == ('Si', 'any.thing')


@pytest.mark.serial
def test_plot_basis(tmp_path):
    basisfile, _ = search_for_file('Ti.dzp.basis')
    pngfile = tmp_path / 'output.png'
    subprocess.check_call(['python', '-m', 'gpaw', '-T', 'plot-basis',
                           basisfile, '--write', pngfile])
    assert pngfile.is_file()
