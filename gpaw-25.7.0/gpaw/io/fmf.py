import os
import time

from gpaw import __version__


class FMF:
    """Full-Metadata Format

    Full-Metadata Format after
    https://www.sir.uni-freiburg.de/repository/2009/SI20090302a/SI20090302a.pdf
    """
    def __init__(self,
                 title='-',
                 creator=None,
                 place=None,
                 escape='#'):
        self.escape = escape
        self.estimate_creator(creator)
        self.title = title
        self.place = place

    def header(self):
        ec = self.escape
        header = ec + '; -*- fmf version: 1.0 -*-\n'
        header += ec + '[*reference]\n'
        header += ec + 'creator: ' + self.creator + '\n'
        header += ec + 'created: ' + time.strftime('%Y-%m-%d %H:%M') + '\n'

        title = self.title
        if isinstance(title, str):
            title = title.split('\n')
        for i, line in enumerate(title):
            if i == 0:
                header += ec + 'title: '
            else:
                header += ec + '       '
            header += line + '\n'

        header += ec + 'gpaw-version: ' + __version__ + '\n'
        u = os.uname()
        header += ec + 'hostname: ' + u.nodename + '\n'
        header += ec + 'architecture: ' + u.machine + '\n'

        return header

    def data(self, definitions):
        ec = self.escape
        data = ec + '[* data definitions]\n'
        for definition in definitions:
            data += ec + definition + '\n'
        data += ec + '[* data]\n'
        return data

    def field(self, title, entries):
        ec = self.escape
        res = ec + '[' + title + ']\n'
        for entry in entries:
            res += ec + entry + '\n'
        return res

    def estimate_creator(self, creator=None):
        self.creator = creator or os.environ.get('USER', 'unknown')
