"""Code for summerschool notebooks."""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def py2ipynb(path: Path,
             kernel: Optional[Dict[str, str]] = None,
             teachermode: Optional[bool] = False) -> None:
    """Convert Python script to ipynb file.

    Hides cells marked with "# teacher" and replaces lines marked with
    "# student: ..."::

        answer = 42  # student: answer = ...

    gives::

        answer = ...

    """
    cells = []
    text = path.read_text()
    assert text.startswith('# %%\n')
    chunks = text[5:].split('\n\n# %%\n')

    for chunk in chunks:
        cell_type = 'code'
        if chunk.startswith(('"""', 'r"""')):
            chunk = chunk.strip('r\n')
            chunk = chunk.strip('"')
            cell_type = 'markdown'

        cell: Dict[str, Any] = {
            'cell_type': cell_type,
            'metadata': {},
            'source': chunk.splitlines(True)}

        if cell_type == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            lines = cell['source']
            for i, line in enumerate(lines):
                if ' # student:' in line and not teachermode:
                    a, b = (x.strip() for x in line.split('# student:'))
                    lines[i] = line.split(a)[0] + b + '\n'
                elif line.startswith('# magic: '):
                    lines[i] = line[9:]
                elif line.lower().startswith('# teacher') and not teachermode:
                    del lines[i:]
                    break

        cells.append(cell)

    outpath = path.with_suffix('.ipynb')
    if kernel is None:
        kernel = {'display_name': 'Python 3',
                  'language': 'python',
                  'name': 'python3'}
    outpath.write_text(
        json.dumps(
            {'cells': cells,
             'metadata': {
                 'kernelspec': kernel,
                 'language_info': {
                     'codemirror_mode': {'name': 'ipython', 'version': 3},
                     'file_extension': '.py',
                     'mimetype': 'text/x-python',
                     'name': 'python',
                     'nbconvert_exporter': 'python',
                     'pygments_lexer': 'ipython3',
                     'version': '3.6.1'}},
             'nbformat': 4,
             'nbformat_minor': 1},
            indent=2))


if __name__ == '__main__':
    py2ipynb(Path(sys.argv[1]))
