import numpy as np


def get_fHxc_Gr(xcflags, rs, q, qF, s2_g):
    xc = xcflags.xc
    if xc == 'rALDA':
        return fHxc_ralda(q, qF)

    elif xcflags.is_apbe:
        return fHxc_apbe(rs, q, s2_g)
    else:
        raise ValueError(f'Unknown xc: {xc}')


def fHxc_ralda(q, qF):
    # rALDA (exchange only) kernel
    # Olsen and Thygesen, Phys. Rev. B 88, 115131 (2013)
    # ALDA up to 2*qF, -vc for q >2qF (such that fHxc vanishes)

    rxalda_A = 0.25
    rxalda_qcut = qF * np.sqrt(1.0 / rxalda_A)

    # construct fHxc(k,r)
    return (0.5 + 0.0j) * (
        (1.0 + np.sign(rxalda_qcut - q[:, np.newaxis])) *
        (1.0 + (-1.0) * rxalda_A * (q[:, np.newaxis] / qF)**2.0))


def fHxc_apbe(rs, q, s2_g):
    # Olsen and Thygesen, Phys. Rev. Lett. 112, 203001 (2014)
    # Exchange only part of the PBE XC kernel, neglecting the terms
    # arising from the variation of the density gradient
    # i.e. second functional derivative
    # d2/drho^2 -> \partial^2/\partial rho^2 at fixed \nabla \rho

    fxc_PBE = get_pbe_fxc(
        pbe_rho=3.0 / (4.0 * np.pi * rs**3.0),
        pbe_s2_g=s2_g)
    rxapbe_qcut = np.sqrt(-4.0 * np.pi / fxc_PBE)

    return (0.5 + 0.0j) * (
        (1.0 + np.sign(rxapbe_qcut - q[:, np.newaxis])) *
        (1.0 + fxc_PBE / (4.0 * np.pi) * (q[:, np.newaxis])**2.0))


def get_fspinHxc_Gr_rALDA(qF, q):
    rxalda_A = 0.25
    rxalda_qcut = qF * np.sqrt(1.0 / rxalda_A)

    fspinHxc_Gr = ((0.5 + 0.0j) *
                   (1.0 + np.sign(rxalda_qcut - q[:, np.newaxis])) *
                   (-1.0) * rxalda_A * (q[:, np.newaxis] / qF)**2.0)
    return fspinHxc_Gr


def get_fspinHxc_Gr_rAPBE(rs, q, s2_g):
    fxc_PBE = get_pbe_fxc(pbe_rho=(3.0 / (4.0 * np.pi * rs**3.0)),
                          pbe_s2_g=s2_g)
    rxapbe_qcut = np.sqrt(-4.0 * np.pi / fxc_PBE)
    fspinHxc_Gr = ((0.5 + 0.0j) *
                   (1.0 + np.sign(rxapbe_qcut - q[:, np.newaxis])) *
                   fxc_PBE / (4.0 * np.pi) * q[:, np.newaxis]**2.0)
    return fspinHxc_Gr


def get_heg_A(rs):
    # Returns the A coefficient, where the
    # q ->0 limiting value of static fxc
    # of the HEG = -\frac{4\pi A }{q_F^2} = f_{xc}^{ALDA}.
    # We need correlation energy per electron and first and second derivs
    # w.r.t. rs
    # See for instance Moroni, Ceperley and Senatore,
    # Phys. Rev. Lett. 75, 689 (1995)
    # (and also Kohn and Sham, Phys. Rev. 140, A1133 (1965) equation 2.7)

    # Exchange contribution
    heg_A = 0.25

    # Correlation contribution
    A_ec, A_dec, A_d2ec = get_pw_lda(rs)
    heg_A += (1.0 / 27.0 * rs**2.0 * (9.0 * np.pi / 4.0)**(2.0 / 3.0) *
              (2 * A_dec - rs * A_d2ec))

    return heg_A


def get_heg_B(rs):
    # Returns the B coefficient, where the
    # q -> oo limiting behaviour of static fxc
    # of the HEG is -\frac{4\pi B}{q^2} - \frac{4\pi C}{q_F^2}.
    # Use the parametrisation of Moroni, Ceperley and Senatore,
    # Phys. Rev. Lett. 75, 689 (1995)

    mcs_xs = np.sqrt(rs)

    mcs_a = (1.0, 2.15, 0.0, 0.435)
    mcs_b = (3.0, 1.57, 0.0, 0.409)

    mcs_num = 0

    for mcs_j, mcs_coeff in enumerate(mcs_a):
        mcs_num += mcs_coeff * mcs_xs**mcs_j

    mcs_denom = 0

    for mcs_j, mcs_coeff in enumerate(mcs_b):
        mcs_denom += mcs_coeff * mcs_xs**mcs_j

    heg_B = mcs_num / mcs_denom
    return heg_B


def get_heg_C(rs):
    # Returns the C coefficient, where the
    # q -> oo limiting behaviour of static fxc
    # of the HEG is -\frac{4\pi B}{q^2} - \frac{4\pi C}{q_F^2}.
    # Again see Moroni, Ceperley and Senatore,
    # Phys. Rev. Lett. 75, 689 (1995)

    C_ec, C_dec, Cd2ec = get_pw_lda(rs)

    heg_C = ((-1.0) * np.pi**(2.0 / 3.0) * (1.0 / 18.0)**(1.0 / 3.0) *
             (rs * C_ec + rs**2.0 * C_dec))

    return heg_C


def get_heg_D(rs):
    # Returns a 'D' coefficient, where the
    # q->0 omega -> oo limiting behaviour
    # of the frequency dependent fxc is -\frac{4\pi D}{q_F^2}
    # see Constantin & Pitarke Phys. Rev. B 75, 245127 (2007) equation 7

    D_ec, D_dec, D_d2ec = get_pw_lda(rs)

    # Exchange contribution
    heg_D = 0.15
    # Correlation contribution
    heg_D += ((9.0 * np.pi / 4.0)**(2.0 / 3.0) * rs / 3.0 *
              (22.0 / 15.0 * D_ec + 26.0 / 15.0 * rs * D_dec))
    return heg_D


def get_pw_lda(rs):
    # Returns LDA correlation energy and its first and second
    # derivatives with respect to rs according to the parametrisation
    # of Perdew and Wang, Phys. Rev. B 45, 13244 (1992)

    pw_A = 0.031091
    pw_alp = 0.21370
    pw_beta = (7.5957, 3.5876, 1.6382, 0.49294)

    pw_logdenom = 2.0 * pw_A * (
        pw_beta[0] * rs**0.5 + pw_beta[1] * rs**1.0 +
        pw_beta[2] * rs**1.5 + pw_beta[3] * rs**2.0)

    pw_dlogdenom = 2.0 * pw_A * (0.5 * pw_beta[0] * rs**(-0.5) +
                                 1.0 * pw_beta[1] +
                                 1.5 * pw_beta[2] * rs**0.5 +
                                 2.0 * pw_beta[3] * rs)

    pw_d2logdenom = 2.0 * pw_A * (-0.25 * pw_beta[0] * rs**(-1.5) +
                                  0.75 * pw_beta[2] * rs**(-0.5) +
                                  2.0 * pw_beta[3])

    pw_logarg = 1.0 + 1.0 / pw_logdenom
    pw_dlogarg = (-1.0) / (pw_logdenom**2.0) * pw_dlogdenom
    pw_d2logarg = 2.0 / (pw_logdenom**3.0) * (pw_dlogdenom**2.0)
    pw_d2logarg += (-1.0) / (pw_logdenom**2.0) * pw_d2logdenom

    # pw_ec = the correlation energy (per electron)
    pw_ec = -2.0 * pw_A * (1 + pw_alp * rs) * np.log(pw_logarg)

    # pw_dec = first derivative

    pw_dec = -2.0 * pw_A * (1 + pw_alp * rs) * pw_dlogarg / pw_logarg
    pw_dec += (-2.0 * pw_A * pw_alp) * np.log(pw_logarg)

    # pw_d2ec = second derivative

    pw_d2ec = (-2.0) * pw_A * pw_alp * pw_dlogarg / pw_logarg
    pw_d2ec += (-2.0) * pw_A * ((1 + pw_alp * rs) *
                                (pw_d2logarg / pw_logarg -
                                 (pw_dlogarg**2.0) / (pw_logarg**2.0)))
    pw_d2ec += (-2.0 * pw_A * pw_alp) * pw_dlogarg / pw_logarg

    return pw_ec, pw_dec, pw_d2ec


def get_pbe_fxc(pbe_rho, pbe_s2_g):
    return get_pbe_fxc_and_intermediate_derivatives(pbe_rho, pbe_s2_g)[0]


def get_pbe_fxc_and_intermediate_derivatives(pbe_rho, pbe_s2_g):
    # The intermediate derivatives are only used for testing.
    pbe_kappa = 0.804
    pbe_mu = 0.2195149727645171

    pbe_denom_g = 1.0 + pbe_mu * pbe_s2_g / pbe_kappa

    F_g = 1.0 + pbe_kappa - pbe_kappa / pbe_denom_g
    Fn_g = -8.0 / 3.0 * pbe_mu * pbe_s2_g / pbe_rho / pbe_denom_g**2.0
    Fnn_g = (-11.0 / 3.0 / pbe_rho * Fn_g -
             2.0 / pbe_kappa * Fn_g**2.0 * pbe_denom_g)

    e_g = -3.0 / 4.0 * (3.0 / np.pi)**(1.0 / 3.0) * pbe_rho**(4.0 / 3.0)
    v_g = 4.0 / 3.0 * e_g / pbe_rho
    f_g = 1.0 / 3.0 * v_g / pbe_rho

    pbe_f_g = f_g * F_g + 2.0 * v_g * Fn_g + e_g * Fnn_g
    return pbe_f_g, F_g, Fn_g, Fnn_g
