"""Tool for generating graps of objects."""
from pathlib import Path
from typing import Any

import ase
import numpy as np
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.new.ase_interface import GPAW
from gpaw.new.brillouin import BZPoints
from gpaw.dft import Parameters


def create_nodes(obj, *objects, include):
    node1 = create_node(obj, include)
    nodes = {node.name: node for node in node1.nodes()}
    for obj in objects:
        for node in create_node(obj, include).nodes():
            if node.name not in nodes:
                nodes[node.name] = node

    for name, node in nodes.items():
        node.subclasses = []
        for cls in node.obj.__class__.__subclasses__():
            if cls.__name__ in nodes:
                node.subclasses.append(nodes[cls.__name__])

    for _ in range(1):
        new = {}
        for name, node in nodes.items():
            bases = node.obj.__class__.__bases__
            cls = bases[0]
            if cls is not object:
                base = nodes.get(cls.__name__)
                base = base or new.get(cls.__name__)
                if base is None:
                    base = Node(node.obj, node.attrs, [], None)
                    base.name = cls.__name__
                    base.has = node.has.copy()
                    new[cls.__name__] = base
                if node not in base.subclasses:
                    base.subclasses.append(node)
                if node is not base:
                    node.base = base

        for name, node in new.items():
            nodes[name] = node

    for node in nodes.values():
        node.fix()

    return list(nodes.values())


def create_node(obj, include):
    attrs = []
    arrows = []
    for key, value in obj.__dict__.items():
        if key[:2] != '__':
            if not include(value):
                attrs.append(key)
            else:
                arrows.append(key)
    return Node(obj, attrs, arrows, include)


class Node:
    def __init__(self, obj, attrs, arrows, include):
        self.obj = obj
        self.name = obj.__class__.__name__
        self.attrs = attrs
        self.has = {key: create_node(getattr(obj, key), include)
                    for key in arrows}
        self.base = None
        self.subclasses = []
        self.rgb = None

    def __repr__(self):
        return (f'Node({self.name}, {self.attrs}, {list(self.has)}, ' +
                f'{self.base.name if self.base is not None else None}, ' +
                f'{[o.name for o in self.subclasses]})')

    def nodes(self):
        yield self
        for node in self.has.values():
            yield from node.nodes()

    def keys(self):
        return set(self.attrs + list(self.has))

    def superclass(self):
        return self if self.base is None else self.base.superclass()

    def fix(self):
        if self.subclasses:
            if len(self.subclasses) > 1:
                keys = self.subclasses[0].keys()
                for node in self.subclasses[1:]:
                    keys &= node.keys()
            else:
                keys = self.keys()
            self.attrs = [attr for attr in self.attrs if attr in keys]
            self.has = {key: value for key, value in self.has.items()
                        if key in keys}
            for obj in self.subclasses:
                obj.attrs = [attr for attr in obj.attrs if attr not in keys]
                obj.has = {key: value for key, value in obj.has.items()
                           if key not in keys}

    def color(self, rgb):
        self.rgb = rgb
        for obj in self.subclasses:
            obj.color(rgb)

    def plot(self, g):
        kwargs = {'style': 'filled',
                  'fillcolor': self.rgb} if self.rgb else {}
        if self.attrs:
            a = r'\n'.join(add_type(attr, getattr(self.obj, attr))
                           for attr in self.attrs)
            txt = f'{{{self.name} | {a}}}'
        else:
            txt = self.name
        g.node(self.name, txt, **kwargs)


def add_type(name: str, obj: Any) -> str:
    type = obj.__class__.__name__
    return f'{name}: {type}'


def plot_graph(figname, nodes, colors={}, replace={}):
    import graphviz
    g = graphviz.Digraph(node_attr={'shape': 'record'})

    for node in nodes:
        if node.name in colors:
            node.color(colors[node.name])

    for node in nodes:
        node.plot(g)
        for key, value in node.has.items():
            key = replace.get(key, key)
            g.edge(node.name, value.superclass().name, label=key)
        if node.base:
            g.edge(node.base.name, node.name, arrowhead='onormal')

    g.render(figname, format='svg')

    try:
        Path(figname).unlink()  # remove "dot" file
    except FileNotFoundError:
        pass


def abc():
    class A:
        def __init__(self, b):
            self.a = 1
            self.b = b

        def m(self):
            pass

    class B:
        pass

    class C(B):
        pass

    nodes = create_nodes(
        A(C()), B(),
        include=lambda obj: obj.__class__.__name__ in 'ABC')
    plot_graph('abc', nodes, {'B': '#ffddff'})


def code():
    fd = GPAW(mode='fd', txt=None)
    pw = GPAW(mode='pw', txt=None)
    lcao = GPAW(mode='lcao', txt=None)
    a = ase.Atoms('H', cell=[2, 2, 2], pbc=1)

    class Atoms:
        def __init__(self, calc):
            self.calc = calc

    a0 = Atoms(fd)
    fd.get_potential_energy(a)
    pw.get_potential_energy(a)
    lcao.get_potential_energy(a)
    ibzwfs = fd.dft.ibzwfs
    ibzwfs.wfs_qs = ibzwfs.wfs_qs[0][0]

    colors = {'BZPoints': '#ddffdd',
              'PotentialCalculator': '#ffdddd',
              'WaveFunctions': '#ddddff',
              'Eigensolver': '#ffffdd',
              'PoissonSolver': '#ffeedd',
              'Hamiltonian': '#eeeeee'}

    def include(obj):
        try:
            mod = obj.__module__
        except AttributeError:
            return False

        return mod.startswith('gpaw.new')

    things = [pw, lcao,
              lcao.dft.ibzwfs.wfs_qs[0][0],
              BZPoints(np.zeros((5, 3)))]
    nodes = create_nodes(a0, *things, include=include)
    plot_graph('code', nodes, colors,
               replace={'wfs_qs': 'wfs_qs[q][s]'})

    # scf.svg:
    nodes = create_nodes(
        fd.dft.density.nct_aX,
        pw.dft.density.nct_aX,
        include=lambda obj: obj.__class__.__name__.startswith('Atom'))
    plot_graph('acf', nodes, {'AtomCenteredFunctions': '#ddffff'})

    # da.svg:
    nodes = create_nodes(
        fd.dft.ibzwfs.wfs_qs.psit_nX,
        pw.dft.ibzwfs.wfs_qs[0][0].psit_nX,
        include=lambda obj:
            getattr(obj, '__module__', '').startswith('gpaw.core') and
            obj.__class__.__name__ != '_lru_cache_wrapper')
    plot_graph('da', nodes, {'DistributedArrays': '#eeeeee',
                             'Domain': '#dddddd'})


def builders():
    b = []
    a = ase.Atoms('H', cell=[2, 2, 2], pbc=1)
    for mode in ['fd', 'pw', 'lcao']:
        b.append(Parameters(mode=mode).dft_component_builder(a))
    nodes = create_nodes(
        *b,
        include=lambda obj:
            obj.__class__.__name__.endswith('Builder') or
            obj.__class__.__name__ == 'InputParameters')
    plot_graph('builder', nodes, {'DFTComponentsBuilder': '#ffeedd'})


def aa():
    nodes = create_nodes(
        AtomArraysLayout([1]).empty(),
        include=lambda obj: obj.__class__.__name__.startswith('Atom'))
    nodes = [node for node in nodes if node.name != 'AtomArrays']
    for node in nodes:
        print(node)
        print(node.has)
        if node.name == 'DistributedArrays':
            node.name = 'AtomArrays'
            node.attrs.remove('dv')
            break
    plot_graph('aa', [node for node in nodes if node.name[0] == 'A'])


def main():
    abc()
    code()
    builders()
    aa()


if __name__ == '__main__':
    main()
