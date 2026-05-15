from __future__ import annotations

import functools
import textwrap
import os
from ast import literal_eval
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING
from xml.dom import minidom
from warnings import warn

import numpy as np

from .. import typing
from ..basis_data import Basis, BasisPlotter
from ..setup_data import SetupData, read_maybe_unzipping, search_for_file
from .aeatom import AllElectronAtom, colors
from .generator2 import (PAWSetupGenerator, parameters,
                         generate, plot_log_derivs)
from .radialgd import AERadialGridDescriptor

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_PartialWaveItem = tuple[int,  # l
                         int,  # n
                         float,  # r_cut
                         float,  # energy
                         typing.Array1D,  # phi_g
                         typing.Array1D]  # phit_g
_ProjectorItem = tuple[int,  # l
                       int,  # n
                       float,  # energy
                       typing.Array1D]  # pt_g


def plot_partial_waves(ax: 'Axes',
                       symbol: str,
                       name: str,
                       rgd: AERadialGridDescriptor,
                       cutoff: float,
                       iterator: Iterable[_PartialWaveItem]) -> None:
    r_g = rgd.r_g
    group_by_l: dict[int, list[_PartialWaveItem]] = {}
    bg_color = _get_patch_color(ax)
    for item in sorted(iterator):
        group_by_l.setdefault(item[0], []).append(item)
    for l, items in group_by_l.items():
        weights = _get_blend_weights(len(items))
        for weight, (_, n, rcut, e, phi_g, phit_g) in zip(weights, items):
            if n == -1:
                gc = rgd.ceil(rcut)
                label = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
            else:
                gc = len(rgd)
                label = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
            color = _blend_colors(colors[l],
                                  background=bg_color,
                                  foreground_weight=weight)
            ax.plot(r_g[:gc], (phi_g * r_g)[:gc], color=color, label=label)
            ax.plot(r_g[:gc], (phit_g * r_g)[:gc], '--', color=color)
    ax.axis(xmin=0, xmax=3 * cutoff)
    ax.set_title(f'Partial waves: {symbol} {name}')
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel(r'$r\phi_{n\ell}(r)$')
    ax.legend()


def plot_projectors(ax: 'Axes',
                    symbol: str,
                    name: str,
                    rgd: AERadialGridDescriptor,
                    cutoff: float,
                    iterator: Iterable[_ProjectorItem]) -> None:
    r_g = rgd.r_g
    group_by_l: dict[int, list[_ProjectorItem]] = {}
    bg_color = _get_patch_color(ax)
    for item in sorted(iterator):
        group_by_l.setdefault(item[0], []).append(item)
    for l, items in group_by_l.items():
        weights = _get_blend_weights(len(items))
        for weight, (_, n, e, pt_g) in zip(weights, items):
            if n == -1:
                label = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
            else:
                label = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
            ax.plot(r_g, pt_g * r_g,
                    color=_blend_colors(colors[l],
                                        background=bg_color,
                                        foreground_weight=weight),
                    label=label)
    ax.axis(xmin=0, xmax=cutoff)
    ax.set_title(f'Projectors: {symbol} {name}')
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel(r'$r\tilde{p}(r)$')
    ax.legend()


def plot_potential_components(ax: 'Axes',
                              symbol: str,
                              name: str,
                              rgd: AERadialGridDescriptor,
                              cutoff: float,
                              components: dict[str, typing.Array1D]) -> None:
    assert components
    radial_grid = rgd.r_g
    for color, (key, label) in zip(
            colors,
            [('xc', 'xc'), ('zero', '0'), ('hartree', 'H'),
             ('pseudo', 'ps'), ('all_electron', 'ae')]):
        if key in components:
            ax.plot(radial_grid, components[key], color=color, label=label)
    arrays = [array for key, array in components.items()
              if key != 'all_electron']
    ax.axis(xmin=0,
            xmax=2 * cutoff,
            ymin=min(array[1:].min() for array in arrays),
            ymax=max(0, *(array[1:].max() for array in arrays)))
    ax.set_title(f'Potential components: {symbol} {name}')
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel('potential [Ha]')
    ax.legend()


def _get_setup_symbol_and_name(setup: SetupData) -> tuple[str, str]:
    return setup.symbol, setup.setupname


def _get_gen_symbol_and_name(gen: PAWSetupGenerator) -> tuple[str, str]:
    aea = gen.aea
    return aea.symbol, aea.xc.name


def _get_setup_cutoff(setup: SetupData) -> float:
    cutoff = setup.r0
    if cutoff is not None:
        return cutoff

    # `.r0` can be `None` for 'old setups', whatever that means
    name = f'{setup.symbol}{setup.Nv}'
    params = parameters[name]
    if len(params) == 3:
        _, radii, extra = params
    else:
        _, radii = params
        extra = {}
    if 'r0' in extra:  # E.g. N5
        value = extra['r0']
        if TYPE_CHECKING:
            assert isinstance(value, float)
        return value
    if not isinstance(radii, Iterable):
        radii = [radii]
    return min(radii)


def _normalize_with_radial_grid(array: typing.Array1D,
                                rgd: AERadialGridDescriptor) -> typing.Array1D:
    result = rgd.zeros()
    result[0] = np.nan
    result[1:] = array[1:] / rgd.r_g[1:]
    return result


def _get_blend_weights(n: int, attenuation: float = .5) -> typing.Array1D:
    return (1 - attenuation) ** np.arange(n)


def _get_patch_color(ax: 'Axes') -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    try:
        color = ax.patch.get_facecolor()
        if color is None:
            color = 'w'
    except AttributeError:
        color = 'w'
    return to_rgb(color)


def _blend_colors(color, background='w', foreground_weight=1.):
    # Too troublesome to type this and refactor to have `mypy`
    # understand what we're doing, not worth it
    from matplotlib.colors import to_rgb

    color = np.array(to_rgb(color))
    background = np.array(to_rgb(background))
    return color * foreground_weight + background * (1 - foreground_weight)


def get_plot_pwaves_params_from_generator(
    gen: PAWSetupGenerator,
) -> tuple[str, str,
           AERadialGridDescriptor, float,
           Iterable[_PartialWaveItem]]:
    return (*_get_gen_symbol_and_name(gen),
            gen.rgd,
            gen.rcmax,
            ((l, n, waves.rcut, e, phi_g, phit_g)
             for l, waves in enumerate(gen.waves_l)
             for n, e, phi_g, phit_g in zip(waves.n_n, waves.e_n,
                                            waves.phi_ng, waves.phit_ng)))


def get_plot_pwaves_params_from_setup(
    setup: SetupData,
) -> tuple[str, str,
           AERadialGridDescriptor, float,
           Iterable[_PartialWaveItem]]:
    return (*_get_setup_symbol_and_name(setup),
            setup.rgd,
            _get_setup_cutoff(setup),
            zip(setup.l_j, setup.n_j, setup.rcut_j, setup.eps_j,
                setup.phi_jg, setup.phit_jg))


def get_plot_projs_params_from_generator(
    gen: PAWSetupGenerator,
) -> tuple[str, str, AERadialGridDescriptor, float, Iterable[_ProjectorItem]]:
    return (*_get_gen_symbol_and_name(gen),
            gen.rgd,
            gen.rcmax,
            ((l, n, e, pt_g)
             for l, waves in enumerate(gen.waves_l)
             for n, e, pt_g in zip(waves.n_n, waves.e_n, waves.pt_ng)))


def get_plot_projs_params_from_setup(
    setup: SetupData,
) -> tuple[str, str, AERadialGridDescriptor, float, Iterable[_ProjectorItem]]:
    return (*_get_setup_symbol_and_name(setup),
            setup.rgd,
            _get_setup_cutoff(setup),
            zip(setup.l_j, setup.n_j, setup.eps_j, setup.pt_jg))


def get_plot_pot_comps_params_from_generator(
    gen: PAWSetupGenerator,
) -> tuple[str, str, AERadialGridDescriptor, float, dict[str, typing.Array1D]]:
    assert gen.vtr_g is not None  # Appease `mypy`

    rgd = gen.rgd
    normalize = functools.partial(_normalize_with_radial_grid, rgd=rgd)
    zero = normalize(gen.v0r_g)
    hartree = normalize(gen.vHtr_g)
    pseudo = normalize(gen.vtr_g)
    all_electron = normalize(gen.aea.vr_sg[0])
    components = {'xc': gen.vxct_g,
                  'zero': zero,
                  'hartree': hartree,
                  'pseudo': pseudo,
                  'all_electron': all_electron}
    return (*_get_gen_symbol_and_name(gen), rgd, gen.rcmax, components)


def get_plot_pot_comps_params_from_setup(
    setup: SetupData,
) -> tuple[str, str, AERadialGridDescriptor, float, dict[str, typing.Array1D]]:
    prefactor = (4 * np.pi) ** -.5
    zero = setup.vbar_g * prefactor
    if setup.vt_g is None:
        pseudo = None
    else:
        pseudo = setup.vt_g * prefactor
    symbol, xc_name = _get_setup_symbol_and_name(setup)
    rgd = setup.rgd
    normalize = functools.partial(_normalize_with_radial_grid, rgd=rgd)

    # Reconstruct the AEA object
    # (Note: this misses the empty bound states from projectors)
    aea = AllElectronAtom(symbol, xc_name,
                          Z=setup.Z,
                          configuration=list(zip(setup.n_j, setup.l_j,
                                                 setup.f_j, setup.eps_j)))
    if setup.has_corehole:
        aea.add(setup.ncorehole, setup.lcorehole, -setup.fcorehole)
    aea.initialize(rgd.N)
    aea.run()
    aea.scalar_relativistic = setup.type == 'scalar-relativistic'
    aea.refine()
    all_electron = normalize(aea.vr_sg[0])
    components = {'zero': zero, 'all_electron': all_electron}

    # FIXME: inconsistent with the `PAWSetupGenerator` results
    # # Re-calculate the XC and Hartree parts
    # from ..xc import XC
    # from .all_electron import calculate_density, calculate_potentials

    # n_g = calculate_density(setup.f_j,
    #                         setup.phi_jg * rgd.r_g[None, :],
    #                         rgd.r_g)
    # n_g += setup.nc_g * prefactor
    # _, vHr_g, xc, _ = calculate_potentials(rgd, XC(xc_name), n_g,
    #                                        setup.Z)
    # hartree = normalize(vHr_g)
    # components.update(xc=xc, hartree=hartree)

    if pseudo is not None:
        components['pseudo'] = pseudo
    return (symbol, xc_name, rgd, _get_setup_cutoff(setup), components)


def reconstruct_paw_gen(setup: SetupData,
                        basis: Basis | None = None) -> PAWSetupGenerator:
    params = {'v0': None, **parse_generator_data(setup.generatordata)}
    gen = generate(**params)
    if basis is not None:
        gen.basis = basis
    return gen


def read_basis_file(basis: str) -> Basis:
    symbol, *chunks, end = os.path.basename(basis).split('.')
    if end == 'gz':
        *chunks, end = chunks
    assert end == 'basis'
    name = '.'.join(chunks)
    return Basis.read_xml(symbol, name, basis)


def read_setup_file(dataset: str) -> SetupData:
    symbol, *name, xc = os.path.basename(dataset).split('.')
    if xc == 'gz':
        *name, xc = name
    setup = SetupData(symbol, xc, readxml=False)
    setup.read_xml(read_maybe_unzipping(dataset))
    if not setup.generatordata:
        generator, = (minidom.parseString(read_maybe_unzipping(dataset))
                      .getElementsByTagName('generator'))
        text, = generator.childNodes
        assert isinstance(text, minidom.Text)
        setup.generatordata = textwrap.dedent(text.data).strip('\n')
    return setup


def parse_generator_data(data: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for line in data.splitlines():
        key, sep, value = line.rstrip(',').partition('=')
        if not (sep and key.isidentifier()):
            continue
        try:
            value = literal_eval(value)
        except Exception:
            continue
        params[key] = value
    return params


def _get_figures_and_axes(
        ngraphs: int,
        separate_figures: bool = False) -> tuple[list['Figure'], list['Axes']]:
    from matplotlib import pyplot as plt

    if separate_figures:
        figs = []
        ax_objs = []
        for _ in range(ngraphs):
            fig = plt.figure()
            figs.append(fig)
            ax_objs.append(fig.gca())
        return figs, ax_objs

    assert ngraphs <= 6, f'Too many plots; expected <= 6, got {ngraphs}'
    if ngraphs > 4:
        layout = 2, 3
    elif ngraphs > 2:
        layout = 2, 2
    else:
        layout = 1, ngraphs

    fig = plt.figure()
    subplots = fig.subplots(*layout).flatten()  # type: ignore
    ntrimmed = layout[0] * layout[1] - ngraphs
    if ntrimmed:
        assert ntrimmed > 0, (f'Too many plots {ngraphs!r} '
                              f'for the layout {layout!r}')
        for ax in subplots[-ntrimmed:]:  # Remove unused subplots
            ax.remove()

    return [fig] * ngraphs, subplots[:ngraphs].tolist()


def plot_dataset(
    setup: SetupData,
    *,
    basis: Basis | None = None,
    gen: PAWSetupGenerator | None = None,
    plot_potential_components: bool = True,
    plot_partial_waves: bool = True,
    plot_projectors: bool = True,
    plot_logarithmic_derivatives: str | None = None,
    separate_figures: bool = False,
    savefig: str | None = None,
) -> tuple[list['Axes'], str | None]:
    """
    Return
    ------
    2-tuple: `tuple[list[Axes], <filename> | None]`
    """
    if gen is not None:
        reconstruct = False
    elif plot_logarithmic_derivatives or plot_potential_components:
        reconstruct = True
    else:
        reconstruct = False
    if reconstruct:
        data = setup.generatordata
        if parse_generator_data(data):
            gen = reconstruct_paw_gen(setup, basis)
        else:
            if data:
                data_status = 'malformed'
            else:
                data_status = 'missing'
            msg = ('cannot reconstruct the `PAWSetupGenerator` object '
                   f'({data_status} `setup.generatordata`), '
                   'so the logarithmic derivatives and/or '
                   '(some of) the potential components cannot be plotted')
            warn(msg, stacklevel=2)
            plot_logarithmic_derivatives = None

    plots: list[Callable] = []

    if gen is None:
        (symbol, name,
         rgd, cutoff, ppw_iter) = get_plot_pwaves_params_from_setup(setup)
        *_, pp_iter = get_plot_projs_params_from_setup(setup)
        *_, pot_comps = get_plot_pot_comps_params_from_setup(setup)
    else:
        # TODO: maybe we can compare the `ppw_iter` and `pp_iter`
        # between the stored and regenerated values for verification
        (symbol, name,
         rgd, cutoff, ppw_iter) = get_plot_pwaves_params_from_generator(gen)
        *_, pp_iter = get_plot_projs_params_from_generator(gen)
        *_, pot_comps = get_plot_pot_comps_params_from_generator(gen)

    if plot_logarithmic_derivatives:
        assert gen is not None
        plots.append(functools.partial(
            plot_log_derivs, gen, plot_logarithmic_derivatives, True))
    if plot_potential_components:
        plots.append(functools.partial(
            # Name clash with local variable
            globals()['plot_potential_components'],
            symbol=symbol, name=name, rgd=rgd, cutoff=cutoff,
            components=pot_comps))
    if plot_partial_waves:
        plots.append(functools.partial(
            # Name clash with local variable
            globals()['plot_partial_waves'],
            symbol=symbol, name=name, rgd=rgd, cutoff=cutoff,
            iterator=ppw_iter))
    if plot_projectors:
        plots.append(functools.partial(
            # Name clash with local variable
            globals()['plot_projectors'],
            symbol=symbol, name=name, rgd=rgd, cutoff=cutoff,
            iterator=pp_iter))

    if basis is not None:
        plots.append(functools.partial(BasisPlotter().plot, basis))

    if savefig is not None:
        separate_figures = False
    figs, ax_objs = _get_figures_and_axes(len(plots), separate_figures)
    assert len(figs) == len(ax_objs) == len(plots)
    for ax, plot_func in zip(ax_objs, plots):
        plot_func(ax=ax)

    if savefig is not None:
        assert len({id(fig) for fig in figs}) == 1
        fig, *_ = figs
        assert fig is not None
        fig.savefig(savefig)

    return ax_objs, savefig


def main(args: SimpleNamespace) -> list['Axes']:
    from matplotlib import pyplot as plt

    if args.search:
        args.dataset, _ = search_for_file(args.dataset)
    setup = read_setup_file(args.dataset)
    sep_figs = args.outfile is None and args.separate_figures
    ax_objs, fname = plot_dataset(
        setup,
        separate_figures=sep_figs,
        plot_potential_components=args.potential_components,
        plot_logarithmic_derivatives=args.logarithmic_derivatives,
        savefig=args.outfile)
    assert ax_objs

    if fname is None:
        plt.show()
    return ax_objs


class CLICommand:
    """Plot the PAW dataset,
    which by default includes the partial waves and the projectors.
    """
    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('-p', '--potential-components',
            action='store_true',
            help='Plot the potential components '
            '(this reconstructs the full PAW setup generator object)')
        add('-l', '--logarithmic-derivatives',
            metavar='spdfg,e1:e2:de,radius',
            help='Plot logarithmic derivatives '
            '(this reconstructs the full PAW setup generator object). '
            'Example: -l spdf,-1:1:0.05,1.3. '
            'Energy range and/or radius can be left out.')
        add('-s', '--separate-figures',
            action='store_true',
            help='If not plotting to a file, '
            'plot the plots in separate figure windows/tabs, '
            'instead of as subplots/panels in the same figure')
        add('-S', '--search',
            action='store_true',
            help='Look into the installed datasets (see `gpaw info`) for the '
            'XML file, instead of treating it as a path')
        add('-o', '--outfile', '--write',
            metavar='FILE',
            help='Write the plots to FILE instead of `plt.show()`-ing them')
        add('dataset',
            metavar='FILE',
            help='XML file from which to read the PAW dataset')

    run = staticmethod(main)
