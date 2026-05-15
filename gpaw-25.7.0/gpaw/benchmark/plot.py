import json
import sys
from collections import defaultdict
from pathlib import Path


def compare(files: list[str]) -> None:
    import matplotlib.pyplot as plt

    paths = [Path(f) for f in files]
    data: defaultdict[
        tuple[str, str, str, str],
        dict[str, tuple[float, float, float]]] = defaultdict(dict)
    versions = []
    refname = f'{paths[0].stem}-new'
    for path in paths:
        for age in ['old', 'new']:
            version = f'{path.stem}-{age}'
            versions.append(version)
            for d in json.loads(path.read_text()):
                if d['calcinfo'] == age:
                    cpus = str(d['mpi-ranks'])
                    if d['mpi-ranks'] <= 4:
                        cpus += 'G'
                    id = (d['shortname'],
                          d['longname'],
                          d['processor'],
                          cpus)
                    data[id][version] = (d['First step'],
                                         d['Second step'],
                                         d['max_rss'])
                    print(id, version)
    versions.sort()
    rows = [['number',
             'name', 'processor', 'cores',
             'step 1 [s]', 'step 2 [s]', 'max rss [GB]']]
    for i, id in enumerate(sorted(data)):
        d = data[id]
        row = [str(i)] + list(id[1:])
        if refname in d:
            t1, t2, m = d[refname]
            row += [f'{t1:.0f}', f'{t2:.0f}', f'{m * 1e-9:.1f}']
            rows.append(row)
    Path('table.csv').write_text(
        '\n'.join(', '.join(row) for row in rows) + '\n')

    X = [id[0] + ((' ' + id[3]) if id[3].endswith('G') else '')
         for id in sorted(data)]
    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    for n, (name, ax) in enumerate(zip(['First step',
                                        'Second step',
                                        'max_rss'],
                                       axs)):
        for version in versions:
            Y = []
            for id in sorted(data):
                d = data[id]
                if version in d and refname in d:
                    y = d[version][n] / d[refname][n] * 100 - 100
                else:
                    y = None
                Y.append(y)
            ax.plot(Y, 'x-', label=version)
        ax.set_ylabel(name + ' [%]')
        if n == 2:
            ax.set_xticks(range(len(X)), X, rotation=70, ha='center')
            ax.legend()
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig('table.png')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    compare(sys.argv[1:])
    plt.show()
