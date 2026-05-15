"""Check that our tab-completion script has been updated."""
import pytest
from ase.cli.completion import update

from gpaw.cli.completion import path
from gpaw.cli.main import commands


@pytest.mark.ci
def test_complete():
    try:
        update(path, commands, test=True)
    except ValueError as e:
        raise ValueError(
            'Please update gpaw/cli/complete.py using '
            '"python3 -m gpaw.test.test_complete".') from e


if __name__ == '__main__':
    update(path, commands, test=False)
