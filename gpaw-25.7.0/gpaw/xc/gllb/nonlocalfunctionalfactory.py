from gpaw.xc.gllb.nonlocalfunctional import NonLocalFunctional


def get_nonlocal_functional(name: str,
                            stencil: int = 2,
                            metallic: bool = False,
                            width: float = None,
                            eps: float = 0.05) -> NonLocalFunctional:
    """Function for building GLLB functionals.

    Parameters:

    name:
        name of the functional
    stencil:
        parameter passed to :class:`~gpaw.xc.gga.GGA`
    metallic:
        parameter passed to :class:`~gpaw.xc.gllb.coefficients.Coefficients`
    width:
        parameter passed to :class:`~gpaw.xc.gllb.coefficients.Coefficients`
    eps:
        parameter passed to :class:`~gpaw.xc.gllb.coefficients.Coefficients`

    Recognized names and implied parameters:
    * GLLB (Contains screening part from B88 functional
            and response part based on simple square root expression
            of orbital energy differences)
    * GLLBM (GLLB with `metallic=True`)
    * GLLBC (GLLB with screening part from PBE + correlation from PBE)
    * GLLBCP86 (GLLB with screening part from B88 + correlation from P86)
    * GLLBSC (GLLB with screening part from PBEsol + correlation from PBEsol)
    * GLLBSCM (GLLBSC with `metallic=True`)
    * GLLBNORESP (Just GLLB screening)
    * GLLBLDA (A test functional, which is just LDA but via
               NonLocalFunctional framework)
    * GLLBPBE (A test functional, which is just PBE but via
               NonLocalFunctional framework)
    """
    from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
    from gpaw.xc.gllb.c_response import C_Response
    from gpaw.xc.gllb.c_xc import C_XC
    from gpaw.xc.gllb.coefficients import Coefficients

    functional = NonLocalFunctional(name)

    scr_functional = None
    xc_functional = None
    response = True
    setup_name = None

    # Set parameters based on the name
    if name == 'GLLB':
        scr_functional = 'GGA_X_B88'
    elif name == 'GLLBM':
        scr_functional = 'GGA_X_B88'
        setup_name = 'GLLB'
        metallic = True
    elif name == 'GLLBSC':
        scr_functional = 'GGA_X_PBE_SOL'
        xc_functional = 'GGA_C_PBE_SOL'
        setup_name = 'GLLBSC'
    elif name == 'GLLBSCM':
        scr_functional = 'GGA_X_PBE_SOL'
        xc_functional = 'GGA_C_PBE_SOL'
        setup_name = 'GLLBSC'
        metallic = True
    elif name == 'GLLBC':
        scr_functional = 'GGA_X_PBE'
        xc_functional = 'GGA_C_PBE'
    elif name == 'GLLBCP86':
        scr_functional = 'GGA_X_B88'
        xc_functional = 'GGA_C_P86'
    elif name == 'GLLBLDA':
        xc_functional = 'LDA'
        response = False
        setup_name = 'LDA'
    elif name == 'GLLBPBE':
        xc_functional = 'PBE'
        response = False
        setup_name = 'PBE'
    elif name == 'GLLBNORESP':
        scr_functional = 'GGA_X_B88'
        response = False
    else:
        raise RuntimeError('Unkown nonlocal density functional: ' + name)

    def func_name_to_dict(name):
        dct = {'name': name}
        if name != 'LDA':
            dct['stencil'] = stencil
        return dct

    # Construct functional
    functional = NonLocalFunctional(name, setup_name=setup_name)
    if scr_functional is not None:
        scr = C_GLLBScr(1.0, functional=func_name_to_dict(scr_functional))
        functional.add_contribution(scr)
    if response:
        coef = Coefficients(eps=eps, width=width)
        resp = C_Response(1.0, coef, metallic=metallic)
        functional.add_contribution(resp)
    if xc_functional is not None:
        xc = C_XC(1.0, functional=func_name_to_dict(xc_functional))
        functional.add_contribution(xc)
    return functional
