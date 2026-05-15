import pytest


@pytest.mark.generate_gpw_files
def test_generate_gpwfiles(all_gpw_files):
    """Dummy test which results in the creation of all gpw file fixtures.

    This test triggers a parametrized fixture which results in
    all gpw files of the gpw_files fixture being accessed.

    See the all_gpw_files fixture."""
