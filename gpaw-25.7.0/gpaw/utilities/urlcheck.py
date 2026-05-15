"""Check URL's in Python files."""
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request

OK = {'https://doi.org/%s',
      'https://arxiv.org/abs/%s',
      'https://gitlab.com/gpaw/gpaw/-/merge_requests/%s',
      'https://gitlab.com/gpaw/gpaw/-/issues/%s',
      'https://xkcd.com/%s',
      'https://gitlab.com/ase/ase.git@master',
      'https://gitlab.com/{name}/{name}.git',
      'https://cmrdb.fysik.dtu.dk/c2db',
      'https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-*.tar.gz',
      'https://wiki.fysik.dtu.dk/gpaw-files',
      'https://wiki.fysik.dtu.dk/gpaw-files/',
      'https://wiki.fysik.dtu.dk/gpaw-files/things/',
      'https://gpaw.readthedocs.io/devel'}

USERAGENT = 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'


def check(root: Path) -> int:
    """Chech URL's in Python files inside root."""
    errors = 0
    for path in root.glob('**/*.py'):
        for n, line in enumerate(path.read_text().splitlines()):
            for url in re.findall(r'https?://\S+', line):
                url = url.rstrip(""",.'"}):""")
                if url not in OK and 'html/_downloads' not in str(path):
                    if '(' in url and ')' not in url:
                        url += ')'
                    if not check1(path, n, url):
                        errors += 1
    return errors


def check1(path: Path, n: int, url: str) -> bool:
    try:
        req = Request(url, headers={'User-Agent': USERAGENT})
        urlopen(req)
    except (HTTPError, URLError, ConnectionResetError) as e:
        print(f'{path}:{n + 1}')
        print(url)
        print(e)
        print()
        return False
    except Exception:
        print(f'{path}:{n + 1}')
        print(url)
        raise
    return True


def test():
    errors = sum(check(Path(f)) for f in ['gpaw', 'doc'])
    assert errors < 10


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        root = Path(arg)
        check(root)
