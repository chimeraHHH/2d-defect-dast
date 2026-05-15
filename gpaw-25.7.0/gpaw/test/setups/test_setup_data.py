import os
import subprocess

import pytest

from gpaw.setup_data import SetupData, read_maybe_unzipping


@pytest.mark.serial
def test_parsing_core_hole_state(in_tmp_dir):
    """
    Test for parsing a core-hole state from an XML file.
    """
    subprocess.run(['gpaw', 'dataset',
                    '--tag', 'mysetup', '--core-hole', '3s,1', '-w', 'Ti'])
    assert os.path.isfile('Ti.mysetup.LDA')
    setup = SetupData('Ti', 'LDA', readxml=False)
    setup.read_xml(read_maybe_unzipping('Ti.mysetup.LDA'))
    assert setup.ncorehole == 3
    assert setup.lcorehole == 0
    assert setup.fcorehole == 1
