r"""
Ascii-art math to LaTeX converter
=================================

Examples:

::

 1+2
 ---
  3

.. math::

  \frac{1+2}{3}

::

 <a|b>

.. math::

  \langle a|b \rangle

::

 /  _ ~ ~    --- a
 | dr 𝜓 𝜓  + >  S
 /     m n   --- ij
             aij

.. math::

  \int d\mathbf{r} \tilde{𝜓}_{m} \tilde{𝜓}_{n}  + \sum^{}_{aij} S^{a}_{ij}
"""

from __future__ import annotations


def prep(lines: list[str], n: int | None) -> tuple[list[str], int | None]:
    """Preprocess lines.

    * Remove leading and trailing empty lines.
    * Make all lines have the same length (pad with spaces).
    * Remove spaces from beginning of lines.
    """
    if not lines:
        return [], n
    while lines and not lines[0].strip():
        lines.pop(0)
        if n is not None:
            n -= 1
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return [], 0
    i = min(len(line) - len(line.lstrip()) for line in lines)
    lines = [line[i:] for line in lines]
    i = max(len(line) for line in lines)
    return [line.ljust(i) for line in lines], n


class ParseError(Exception):
    """Bad ascii-art math."""


def cut(lines: list[str], i1: int, i2: int = None) -> list[str]:
    """Cut out block.

    >>> cut(['012345', 'abcdef'], 1, 3)
    ['12', 'bc']
    """
    index = slice(i1, i2)
    return [line[index] for line in lines]


def block(lines: list[str]) -> dict[int, str]:
    r"""Find superscript/subscript blocks.

    >>> block([' 2  _ ',
    ...        '    k '])
    {0: '2', 3: '\\mathbf{k}'}
    """
    if not lines:
        return {}
    blocks = {}
    i1 = None
    for i in range(len(lines[0])):
        if all(line[i] == ' ' for line in lines):
            if i1 is not None:
                blocks[i1 - 1] = parse(cut(lines, i1, i))
                i1 = None
        else:
            if i1 is None:
                i1 = i
    if i1 is not None:
        blocks[i1 - 1] = parse(cut(lines, i1))
    return blocks


def parse(lines: str | list[str], n: int = None) -> str:
    r"""Parse ascii-art math to LaTeX.

    >>> parse([' /   ~      ',
    ...        ' |dx p  (x) ',
    ...        ' /    ai    '])
    '\\int dx \\tilde{p}_{ai}  (x)'
    >>> parse(['   _ _ ',
    ...        '  ik.r ',
    ...        ' e     '])
    'e^{i\\mathbf{k}\\cdot \\mathbf{r}}'
    """
    if isinstance(lines, str):
        lines = lines.splitlines()

    lines, n = prep(lines, n)

    if not not False:
        print('*********************************************')
        print(n)
        print('\n'.join(lines))

    if not lines:
        return ''
    if n is None:
        N = max((len(line.replace(' ', '')), n)
                for n, line in enumerate(lines))[1]
        for n in [N] + [n for n in range(len(lines)) if n != N]:
            try:
                latex = parse(lines, n)
            except ParseError:
                continue
            return latex
        raise ParseError('Could not parse\n\n' + '    \n'.join(lines))

    line = lines[n]
    i1 = line.find('--')
    if i1 != -1:
        i2 = len(line) - len(line[i1:].lstrip('-'))
        p1 = parse(cut(lines, 0, i1), n)
        p2 = parse(cut(lines[:n], i1, i2))
        p3 = parse(cut(lines[n + 1:], i1, i2))
        p4 = parse(cut(lines, i2), n)
        return rf'{p1} \frac{{{p2}}}{{{p3}}} {p4}'.strip()

    i = line.find('>')
    if i != -1 and n > 0 and lines[n - 1][i] == '-':
        line1 = lines[n - 1]
        i2 = len(line1) - len(line1[i:].lstrip('-'))
        p1 = parse(cut(lines, 0, i), n)
        p2 = parse(cut(lines[:n - 1], i, i2))
        p3 = parse(cut(lines[n + 2:], i, i2))
        p4 = parse(cut(lines, i2), n)
        return rf'{p1} \sum^{{{p2}}}_{{{p3}}} {p4}'.strip()

    i = line.find('|')
    if i != -1:
        if n > 0 and lines[n - 1][i] == '/':
            p1 = parse(cut(lines, 0, i), n)
            p2 = parse(cut(lines, i + 1), n)
            return rf'{p1} \int {p2}'.strip()
        i1 = line.find('<')
        i2 = line.find('>')
        if i1 != -1 and i1 <= i and i2 != -1 and i2 >= i:
            p1 = parse(cut(lines, 0, i1), n)
            p2 = parse(cut(lines, i1 + 1, i), n)
            p3 = parse(cut(lines, i + 1, i2), n)
            p4 = parse(cut(lines, i2 + 1), n)
            return rf'{p1} \langle {p2}|{p3} \rangle {p4}'.strip()

    hats = {}
    if n > 0:
        new = []
        for i, c in enumerate(lines[n - 1]):
            if c in '^~_':
                hats[i] = c
                c = ' '
            new.append(c)
        lines[n - 1] = ''.join(new)

    superscripts = block(lines[:n])
    subscripts = block(lines[n + 1:])

    results = []
    for i, c in enumerate(line):

        if i in hats:
            hat = {'^': 'hat', '~': 'tilde', '_': 'mathbf'}[hats[i]]
            c = rf'\{hat}{{{c}}}'
        sup = superscripts.pop(i, None)
        if sup:
            c = rf'{c}^{{{sup}}}'
        sub = subscripts.pop(i, None)
        if sub:
            c = rf'{c}_{{{sub}}}'
        c = {'.': r'\cdot '}.get(c, c)
        results.append(c)

    if superscripts or subscripts:
        raise ParseError(f'super={superscripts}, sub={subscripts}')

    latex = ''.join(results).strip()

    for sequence, replacement in [
        ('->', r'\rightarrow'),
        ('<-', r'\leftarrow')]:
        latex = latex.replace(sequence, replacement)

    return latex


def autodoc_process_docstring(lines):
    """Hook-function for Sphinx."""
    blocks = []
    for i1, line in enumerate(lines):
        if line.endswith(':::'):
            for i2, line in enumerate(lines[i1 + 2:], i1 + 2):
                if not line:
                    break
            else:
                i2 += 1
            blocks.append((i1, i2))
    for i1, i2 in reversed(blocks):
        latex = parse(lines[i1 + 1:i2])
        line = f'.. math:: {latex}'
        if lines[i1].strip() == ':::':
            lines[i1:i2] = [line]
        else:
            lines[i1:i2] = [lines[i1][:-2], '', line]


def test_examples():
    """Test examples from module docstring."""
    lines = __doc__.replace('\n::', '\n:::').splitlines()
    autodoc_process_docstring(lines)
    for example in '\n'.join(lines).split('.. math:: ')[1:]:
        line1, *lines = example.splitlines()
        line2 = lines[3].strip()
        assert line1 == line2


def main():
    import sys
    import argparse
    import importlib
    parser = argparse.ArgumentParser(
        description='Parse docstring with ascii-art math.')
    parser.add_argument(
        'thing',
        nargs='?',
        help='Name of module, class, method or '
        'function.  Examples: "module.submodule", '
        '"module.Class", "module.Class.method", '
        '"module.function".  Will read from stdin if not given.')
    args = parser.parse_args()
    if args.thing is None:
        lines = sys.stdin.read().splitlines()
        print(parse(lines))
    else:
        parts = args.thing.split('.')
        for i, part in enumerate(parts):
            if not part.islower():
                mod = importlib.import_module('.'.join(parts[:i]))
                break
        else:
            # no break
            try:
                mod = importlib.import_module('.'.join(parts))
                i += 1
            except ImportError:
                mod = importlib.import_module('.'.join(parts[:-1]))
        thing = mod
        for part in parts[i:]:
            thing = getattr(thing, part)
        lines = thing.__doc__.splitlines()
        autodoc_process_docstring(lines)
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
