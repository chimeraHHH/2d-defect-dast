"""Create softlinks to JTH datasets."""
import re
import sys
from pathlib import Path


if __name__ == '__main__':
    for path in Path(sys.argv[1]).glob('*.*-JTH.xml'):
        m = re.match(r'([A-Z][a-z]?)\.[A-Z]+_([A-Z]+)-JTH\.xml', path.name)
        assert m is not None
        symbol = m[1]
        xc = m[2]
        Path(f'{symbol}.jth.{xc}').symlink_to(path)
