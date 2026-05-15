import pytest
import numpy as np

from gpaw.response.tool import (get_bz_transitions,
                                get_chi0_integrand,
                                get_degeneracy_matrix,
                                get_individual_transition_strengths)


@pytest.mark.response
def test_response_pdens_tool(in_tmp_dir, gpw_files):
    """Calculate optical transition strengths."""
    spins = 'all'
    q_c = [0., 0., 0.]
    bzk_kc = np.array([[0., 0., 0.]])

    pair, qpd, domain = get_bz_transitions(
        gpw_files['silicon_pdens_tool'], q_c, bzk_kc, spins=spins, ecut=10)

    nocc1, nocc2 = pair.gs.count_occupied_bands(1e-6)
    # XXX should we know 1e-6?

    # non-empty bands
    n_n = np.arange(0, nocc2)
    # not completely filled bands
    m_m = np.arange(nocc1, pair.gs.bd.nbands)

    nt = len(domain)
    nn = len(n_n)
    nm = len(m_m)
    nG = qpd.ngmax
    optical_limit = np.allclose(q_c, 0.)

    n_tnmG = np.zeros((nt, nn, nm, nG + 2 * optical_limit), dtype=complex)
    df_tnm = np.zeros((nt, nn, nm), dtype=float)
    eps_tn = np.zeros((nt, nn), dtype=float)
    eps_tm = np.zeros((nt, nm), dtype=float)

    for t, point in enumerate(domain):
        (n_tnmG[t], df_tnm[t],
         eps_tn[t], eps_tm[t]) = get_chi0_integrand(pair, qpd,
                                                    n_n, m_m, point)

    testNM_ibN = [[[0], [4, 5, 6]], [[0], [7]],
                  [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [7]]]
    testts_iG = np.array([[0.07, 0.07, 0.07], [0.00, 0.00, 0.00],
                          [51.34, 51.34, 51.34], [22.69, 22.69, 22.69]])

    for t, (point, n_nmG,
            df_nm, eps_n, eps_m) in enumerate(zip(domain, n_tnmG, df_tnm,
                                                  eps_tn, eps_tm)):

        # Find degeneracies
        degmat_Nn, eps_N = get_degeneracy_matrix(eps_n, tol=1.e-2)
        degmat_Mm, eps_M = get_degeneracy_matrix(eps_m, tol=1.e-2)

        # Find diagonal transition strengths
        its_nmG = np.zeros((nn, nm, 1 + 2 * optical_limit))
        for G in range(1 + 2 * optical_limit):
            its_nmG[:, :, G] = get_individual_transition_strengths(
                n_nmG, df_nm,
                G, G)

        # Find unique transition strengths
        its_NmG = np.tensordot(degmat_Nn, its_nmG, axes=(1, 0))
        ts_MNG = np.tensordot(degmat_Mm, its_NmG, axes=(1, 1))
        ts_NMG = np.transpose(ts_MNG, (1, 0, 2))

        i = 0
        for N, ts_MG in enumerate(ts_NMG):
            for M, ts_G in enumerate(ts_MG):
                degN_n = n_n[np.where(degmat_Nn[N])]
                degM_m = m_m[np.where(degmat_Mm[M])]

                for testn, n in zip(testNM_ibN[i][0], degN_n):
                    assert testn == pytest.approx(n, abs=0.5)
                for testm, m in zip(testNM_ibN[i][1], degM_m):
                    assert testm == pytest.approx(m, abs=0.5)

                for testts, ts in zip(testts_iG[i], ts_G):
                    print(ts)
                    assert testts == pytest.approx(ts, abs=0.01)

                i += 1
