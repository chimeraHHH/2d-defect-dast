from .wrapper import HybridXC

__all__ = ['HybridXC']


def parse_name(name: str) -> tuple[str, float, float, bool]:
    """Parse known hybrid functional names.

    Returns:

    * libxc-name of semi-local functional
    * exact-exchange fraction
    * damping coefficient
    * Yukawa screening

    >>> parse_name('YS-PBE0')
    ('GGA_X_SFAT_PBE', 0.25, 0.165, True)
    """
    if name == 'EXX':
        return 'null', 1.0, 0.0, False
    if name == 'PBE0':
        return 'HYB_GGA_XC_PBEH', 0.25, 0.0, False
    if name == 'HSE03':
        return 'HYB_GGA_XC_HSE03', 0.25, 0.106, False
    if name == 'HSE06':
        return 'HYB_GGA_XC_HSE06', 0.25, 0.11, False
    if name == 'B3LYP':
        return 'HYB_GGA_XC_B3LYP', 0.2, 0.0, False
    if name == 'YS-PBE0':
        return 'GGA_X_SFAT_PBE', 0.25, 1.5 * 0.11, True
    raise ValueError(f'Unknown hybrid functional: {name}')
