import os
import fnmatch
from io import BytesIO, StringIO
import tarfile
import re
from urllib.request import urlopen
import ssl
import contextlib
import shlex


sources = [('gpaw', 'official GPAW setups releases'),
           ('sg15', 'SG15 pseudopotentials'),
           ('basis', 'basis sets for LCAO mode'),
           ('test', 'small file for testing this script')]

names = [r for r, d in sources]


# (We would like to use https always, but quantum-simulation.org does
# not support that as of 2025-02-03)
baseurls = {
    'gpaw': 'https://gitlab.com/gpaw/gpaw/-/raw/master/doc/setups/setups.rst',
    'sg15': 'http://www.quantum-simulation.org/potentials/sg15_oncv/',
    'basis': 'https://wiki.fysik.dtu.dk/gpaw-files/',
    'test': 'https://wiki.fysik.dtu.dk/gpaw-files/'}


notfound_msg = """\
For some reason the files were not found.

Perhaps this script is out of date, and the data is no longer
available at the expected URL:

  {url}

Or maybe there it is just a temporary problem or timeout.  Please try
again, or rummage around the GPAW web page until a solution is found.
Writing e-mails to gpaw-users@listserv.fysik.dtu.dk or reporting
an issue on https://gitlab.com/gpaw/gpaw/issues is also
likely to help."""


def urlopen_nocertcheck(src):
    """Open a URL on a server without checking the certificate.

    Some data is read from a DTU server with a self-signed
    certificate.  That causes trouble on some machines.
    """

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urlopen(src, context=ctx)


class CLICommand:
    """Install additional PAW datasets, pseudopotential or basis sets.

    Without a directory or a source flag, show available setups and GPAW
    setup paths.

    With a directory and a source flag, download and install gpaw-setups
    into INSTALLDIR/[setups-package-name-and-version].
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('directory', nargs='?')
        add('--version',
            help='download VERSION of package.  '
            'Run without arguments to display a list of versions.  '
            'VERSION can be the full URL or a part such as  '
            '\'0.8\' or \'0.6.6300\'')
        add('--list-all', action='store_true',
            help='list packages from all sources')
        src_group = (parser
                     .add_argument_group('source flags')
                     .add_mutually_exclusive_group())
        src_add = src_group.add_argument
        src_add('--tarball', metavar='FILE',
                help='unpack and install from local tarball FILE '
                'instead of downloading')
        for name, help in sources:
            src_add('--' + name, action='store_const',
                    const=name, dest='source',
                    help=help)
        reg_group = (parser
                     .add_argument_group('registration flags (script runs '
                                         'interactively if neither is '
                                         'supplied)')
                     .add_mutually_exclusive_group())
        reg_add = reg_group.add_argument
        reg_add('--register', action='store_const', const=True,
                help='run non-interactively and register install path in '
                'GPAW setup search paths.  This is done by adding lines to '
                '~/.gpaw/rc.py')
        reg_add('--no-register', action='store_const',
                const=False, dest='register',
                help='run non-interactively and do not register install path '
                'in GPAW setup search paths')

    @staticmethod
    def run(args, parser):
        main(args, parser)


def main(args, parser):
    if args.source is None:
        args.list_all = True

    def print_blurb():
        print_setups_info(parser)
        print()
        print('Run gpaw install-data --SOURCE DIR to install the newest '
              'setups into DIR.')
        print('Run gpaw install-data --SOURCE --version=VERSION DIR to '
              'install VERSION (from above).')
        print('See gpaw install-data --help for more info.')

    # The sg15 file is a tarbomb.  We will later defuse it by untarring
    # into a subdirectory, so we don't leave a ghastly mess on the
    # unsuspecting user's system.

    if not args.tarball:
        if args.list_all:
            urls_dict = {source: get_urls(source) for source in names}
        else:
            urls_dict = {args.source: get_urls(args.source)}

        def print_urls(urls, marked=None, file=None):
            for url in urls:
                pageurl, fname = url.rsplit('/', 1)
                if url == marked:
                    marking = ' [*]'
                else:
                    marking = '    '
                print(f' {marking} {url}', file=file)

        def print_all_urls(source=None, marked=None, file=None):
            if source:
                displayed_urls = {source: urls_dict[source]}
            else:
                displayed_urls = urls_dict
            for source, url_sublist in displayed_urls.items():
                print(f'Available setups and pseudopotentials (--{source}):',
                      file=file)
                print_urls(url_sublist, marked, file)
                print(file=file)

        if args.source:
            urls = urls_dict[args.source]
        else:
            print_all_urls()
            print_blurb()
            raise SystemExit

        if len(urls) == 0:
            url = baseurls[args.source]
            parser.error(notfound_msg.format(url=url))

        if args.version:
            matching_urls = [url for url in urls
                             if match_version(url, args.version)]
            with StringIO() as fobj:
                if len(matching_urls) > 1:
                    print('\nMore than one setup file matches version '
                          '"%s":' % args.version,
                          file=fobj)
                    print_urls(matching_urls, file=fobj)
                elif len(matching_urls) == 0:
                    print('\nNo setup matched the specified version '
                          '"%s".' % args.version,
                          file=fobj)
                    print_all_urls(args.source, file=fobj)
                error_msg = fobj.getvalue()
                if error_msg:
                    parser.error(error_msg)
            url, = matching_urls
        else:
            url = urls[0]

        print_all_urls(marked=url)

    if not args.directory:
        print_blurb()
        raise SystemExit

    targetpath = args.directory

    with contextlib.ExitStack() as stack:
        push = stack.enter_context
        if args.tarball:
            print('Reading local tarball %s' % args.tarball)
            targzfile = push(tarfile.open(args.tarball))
            tarfname = args.tarball
        else:
            tarfname = url.rsplit('/', 1)[1]
            print('Selected %s.  Downloading...' % tarfname)
            response = push(urlopen_nocertcheck(url))
            resp_fobj = push(BytesIO(response.read()))
            targzfile = push(tarfile.open(fileobj=resp_fobj))

        if not os.path.exists(targetpath):
            os.makedirs(targetpath)

        assert tarfname.endswith('.tar.gz')
        # remove .tar.gz ending
        setup_dirname = tarfname.rsplit('.', 2)[0]
        setup_path = os.path.abspath(os.path.join(targetpath,
                                                  setup_dirname))
        if tarfname.startswith('sg15'):
            # Defuse tarbomb
            if not os.path.isdir(setup_path):
                os.mkdir(setup_path)
            targetpath = os.path.join(targetpath, setup_dirname)

        print('Extracting tarball into %s' % targetpath)
        targzfile.extractall(targetpath)
        assert os.path.isdir(setup_path)
        print('Setups installed into %s.' % setup_path)

    # Okay, now we have to maybe edit people's rc files.
    rcfiledir = os.path.join(os.environ['HOME'], '.gpaw')
    rcfilepath = os.path.join(rcfiledir, 'rc.py')

    # We could do all this by importing the rcfile as well and checking
    # whether things are okay or not.
    rcline = "setup_paths.insert(0, {!r})".format(setup_path)

    # Run interactive mode unless someone specified a flag requiring otherwise
    interactive_mode = args.register is None

    register_path = False

    if interactive_mode:
        answer = input('Register this setup path in %s? [y/n] ' % rcfilepath)
        if answer.lower() in ['y', 'yes']:
            register_path = True
        elif answer.lower() in ['n', 'no']:
            print('As you wish.')
        else:
            print('What do you mean by "%s"?  Assuming "n".' % answer)
    else:
        register_path = args.register

    if register_path:
        # First we create the file
        if not os.path.exists(rcfiledir):
            os.makedirs(rcfiledir)
        if not os.path.exists(rcfilepath):
            with open(rcfilepath, 'w'):  # Just create empty file
                pass

        with open(rcfilepath) as fobj:
            for line in fobj:
                if line.startswith(rcline):
                    print('It looks like the path is already registered in %s.'
                          % rcfilepath)
                    print('File will not be modified at this time.')
                    break
            else:
                with open(rcfilepath, 'a') as rcfd:
                    print(rcline, file=rcfd)
                print('Setup path registered in %s.' % rcfilepath)

                print_setups_info(parser)
    else:
        print('You can manually register the setups by adding the')
        print('following line to %s:' % rcfilepath)
        print()
        print(rcline)
        print()
        print('Or if you prefer to use environment variables, you can')
        print('set GPAW_SETUP_PATH. For example:')
        print()
        print(f'export GPAW_SETUP_PATH={shlex.quote(setup_path)}')
        print()
    print('Installation complete.')


def get_urls(source):
    page = baseurls[source]
    if source == 'gpaw':
        with urlopen_nocertcheck(page) as response:
            pattern = ('https://wiki.fysik.dtu.dk/gpaw-files/'
                       'gpaw-setups-*.tar.gz')
            lines = (line.strip().decode() for line in response)
            urls = [line for line in lines if fnmatch.fnmatch(line, pattern)]

    elif source == 'sg15':
        # We want sg15_oncv_2015-10-07.tar.gz, but they may upload
        # newer files, too.
        pattern = (r'<a\s*href=[^>]+>\s*'
                   r'(sg15_oncv_upf_\d\d\d\d-\d\d-\d\d.tar.gz)'
                   r'\s*</a>')

        with urlopen_nocertcheck(page) as response:
            txt = response.read().decode('ascii', errors='replace')
        files = re.compile(pattern).findall(txt)
        files.sort(reverse=True)
        urls = [page + fname for fname in files]

    elif source == 'basis':
        files = ['gpaw-basis-NAO-sz+coopt-NGTO-0.9.11271.tar.gz',
                 'gpaw-basis-pvalence-0.9.11271.tar.gz',
                 'gpaw-basis-pvalence-0.9.20000.tar.gz']
        urls = [page + fname for fname in files]

    elif source == 'test':
        urls = [page + 'gpaw-dist-test-source.tar.gz']

    else:
        raise ValueError('Unknown source: %s' % source)

    return urls


def print_setups_info(parser):
    try:
        import gpaw
    except ImportError as e:
        parser.error('Cannot import \'gpaw\'.  GPAW does not appear to be '
                     'installed. %s' % e)

    # The contents of the rc file may have changed.  Thus, we initialize
    # setup_paths again to be sure that everything is as it should be.
    gpaw.setup_paths[:] = gpaw.standard_setup_paths()
    gpaw.read_rc_file()
    gpaw.initialize_data_paths()

    npaths = len(gpaw.setup_paths)
    if npaths == 0:
        print('GPAW currently has no setup search paths')
    else:
        print('Current GPAW setup paths in order of search priority:')
        for i, path in enumerate(gpaw.setup_paths):
            print('%4d. %s' % (i + 1, path))


def get_runs(seq, criterion=lambda x: x):
    """
    >>> get_runs('aaabacbbcab')  # doctest: +NORMALIZE_WHITESPACE
    [['a', 'a', 'a'], ['b'], ['a'], ['c'], ['b', 'b'], ['c'], ['a'],
     ['b']]
    >>> get_runs('foo,bar,baz', str.isalnum)
    [['f', 'o', 'o'], [','], ['b', 'a', 'r'], [','], ['b', 'a', 'z']]
    >>> get_runs(  # doctest: +NORMALIZE_WHITESPACE
    ...     [1, 2, 5, 3, 4, 7, 10, 3, 5, 2], lambda x: x % 3,
    ... )
    [[1], [2, 5], [3], [4, 7, 10], [3], [5, 2]]
    """
    if not seq:
        return []
    runs = []
    for item in seq:
        value = criterion(item)
        try:
            if value == runs[-1][0]:
                runs[-1][1].append(item)
                continue
        except IndexError:  # Empty `runs`
            pass
        runs.append((value, [item]))
    return [run for _, run in runs]


def split_into_chunks(string):
    """
    >>> split_into_chunks('')
    []
    >>> split_into_chunks('foo')
    ['foo']
    >>> split_into_chunks('version 10.1.rc0')
    ['version', ' ', '10', '.', '1', '.', 'rc0']
    >>> split_into_chunks('https://gpaw.readthedocs.io/')
    ['https', '://', 'gpaw', '.', 'readthedocs', '.', 'io', '/']
    """
    return [''.join(run) for run in get_runs(string, str.isalnum)]


def match_version(url, version):
    """
    >>> match_version('0.9.0', '0')
    True
    >>> match_version('1.9.0', '0')
    False
    >>> match_version('foo.0', '.0')
    True
    >>> match_version('1.0', '.0')
    False
    >>> match_version('11', '1')
    False
    >>> match_version('foo.bar.1.0', 'bar.1')
    True
    >>> match_version('foo.bar-1.0', 'bar.1')
    False
    >>> match_version('foobar.1', 'bar.1')
    False
    >>> match_version('foo.bar.11', 'bar.1')
    False
    """
    url_chunks = split_into_chunks(url)
    version_chunks = split_into_chunks(version)
    num_chunks = len(version_chunks)
    try:
        token_offset = 0 if version_chunks[0].isalnum() else 1
        first_token_is_numeric = version_chunks[token_offset].isnumeric()
    except IndexError:
        raise ValueError(
            f'version = {version!r}: cannot find any alphanumeric token'
        ) from None
    # A match starting from the beginning is always a match
    if version_chunks == url_chunks[:num_chunks]:
        return True
    # A match against the middle of the string may be a false positive
    for rolling_offset in range(1, len(url_chunks) - num_chunks + 2):
        # Non-match
        if version_chunks != url_chunks[rolling_offset:][:num_chunks]:
            is_match = False
        # 'bar.0' matchs 'foo.bar.0' and '1.bar.0'
        elif not first_token_is_numeric:
            is_match = True
        # '0' matches 'foo.0' but not '1.0'
        else:
            previous_token = url_chunks[rolling_offset + token_offset - 2]
            is_match = not previous_token.isnumeric()
        if is_match:
            return True
    return False
