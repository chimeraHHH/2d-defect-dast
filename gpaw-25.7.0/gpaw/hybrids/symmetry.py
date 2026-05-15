from math import pi
from collections import defaultdict

import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from .kpts import RSKPoint, to_real_space


def create_symmetry_map(kd: KPointDescriptor):  # -> List[List[int]]
    sym = kd.symmetry
    U_scc = sym.op_scc
    nsym = len(U_scc)
    compconj_s = np.zeros(nsym, bool)
    if sym.time_reversal and not sym.has_inversion:
        U_scc = np.concatenate([U_scc, -U_scc])
        compconj_s = np.zeros(nsym * 2, bool)
        compconj_s[nsym:] = True
        nsym *= 2

    map_ss = np.zeros((nsym, nsym), int)
    for s1 in range(nsym):
        for s2 in range(nsym):
            diff_s = abs(U_scc[s1].dot(U_scc).transpose((1, 0, 2)) -
                         U_scc[s2]).sum(2).sum(1)
            indices = (diff_s == 0).nonzero()[0]
            assert len(indices) == 1
            s = indices[0]
            assert compconj_s[s1] ^ compconj_s[s2] == compconj_s[s]
            map_ss[s1, s2] = s

    return map_ss


class Symmetry:
    def __init__(self, kd: KPointDescriptor):
        self.kd = kd
        self.symmetry_map_ss = create_symmetry_map(kd)

        U_scc = kd.symmetry.op_scc
        is_identity_s = (U_scc == np.eye(3, dtype=int)).all(2).all(1)
        self.s0 = is_identity_s.nonzero()[0][0]
        self.inverse_s = self.symmetry_map_ss[:, self.s0]

    def symmetry_operation(self, s: int, wfs, inverse=False):
        if inverse:
            s = self.inverse_s[s]
        U_scc = self.kd.symmetry.op_scc
        nsym = len(U_scc)
        time_reversal = s >= nsym
        s %= nsym
        U_cc = U_scc[s]

        if (U_cc == np.eye(3, dtype=int)).all():
            def T0(a_R):
                return a_R
        else:
            N_c = wfs.gd.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')

            def T0(a_R):
                return a_R.ravel()[i].reshape(N_c)

        if time_reversal:
            def T(a_R):
                return T0(a_R).conj()
        else:
            T = T0

        T_a = []
        for a, id in enumerate(wfs.setups.id_a):
            b = self.kd.symmetry.a_sa[s, a]
            S_c = np.dot(wfs.spos_ac[a], U_cc) - wfs.spos_ac[b]
            U_ii = wfs.setups.atomrotations.get_by_a(a).R_sii[s].T
            T_a.append((b, S_c, U_ii))

        return T, T_a, time_reversal

    def apply_symmetry(self, s: int, rsk, wfs, spos_ac):
        U_scc = self.kd.symmetry.op_scc
        nsym = len(U_scc)
        time_reversal = s >= nsym
        s %= nsym
        sign = 1 - 2 * int(time_reversal)
        U_cc = U_scc[s]

        if (U_cc == np.eye(3)).all() and not time_reversal:
            return rsk

        u1_nR = rsk.u_nR
        proj1 = rsk.proj
        f_n = rsk.f_n
        k1_c = rsk.k_c
        weight = rsk.weight

        u2_nR = np.empty_like(u1_nR)
        proj2 = proj1.new()

        k2_c = sign * U_cc.dot(k1_c)

        N_c = u1_nR.shape[1:]
        i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
        i = np.ravel_multi_index(i_cr, N_c, 'wrap')
        for u1_R, u2_R in zip(u1_nR, u2_nR):
            u2_R[:] = u1_R.ravel()[i].reshape(N_c)

        for a, id in enumerate(wfs.setups.id_a):
            b = self.kd.symmetry.a_sa[s, a]
            S_c = np.dot(spos_ac[a], U_cc) - spos_ac[b]
            x = np.exp(2j * pi * np.dot(k1_c, S_c))
            U_ii = wfs.setups.atomrotations.get_by_a(a).R_sii[s].T * x
            proj2[a][:] = proj1[b].dot(U_ii)

        if time_reversal:
            np.conj(u2_nR, out=u2_nR)
            np.conj(proj2.array, out=proj2.array)

        return RSKPoint(u2_nR, proj2, f_n, k2_c, weight)

    def pairs(self, kpts, wfs, spos_ac):
        kd = self.kd
        nsym = len(kd.symmetry.op_scc)

        assert len(kpts) == kd.nibzkpts

        symmetries_k = []
        for k in range(kd.nibzkpts):
            indices = np.where(kd.bz2ibz_k == k)[0]
            sindices = (kd.sym_k[indices] +
                        kd.time_reversal_k[indices] * nsym)
            symmetries_k.append(sindices)

        # pairs: Dict[Tuple[int, int, int], int]

        pairs1 = defaultdict(int)
        for i1 in range(kd.nibzkpts):
            for s1 in symmetries_k[i1]:
                for i2 in range(kd.nibzkpts):
                    for s2 in symmetries_k[i2]:
                        s3 = self.symmetry_map_ss[s1, s2]
                        # s3 = self.inverse_s[s3]
                        if 1:  # i1 < i2:
                            pairs1[(i1, i2, s3)] += 1
                        else:
                            s4 = self.inverse_s[s3]
                            if i1 == i2:
                                # pairs1[(i1, i1, min(s3, s4))] += 1
                                pairs1[(i1, i1, s3)] += 1
                            else:
                                pairs1[(i2, i1, s4)] += 1
        pairs = {}
        seen = {}
        for (i1, i2, s), count in pairs1.items():
            k2 = kd.bz2bz_ks[kd.ibz2bz_k[i2], s]
            if (i1, k2) in seen:
                pairs[seen[(i1, k2)]] += count
            else:
                pairs[(i1, i2, s)] = count
                # seen[(i1, k2)] = (i1, i2, s)

        comm = wfs.world
        lasti1 = -1
        lasti2 = -1
        for (i1, i2, s), count in sorted(pairs.items()):
            if i1 != lasti1:
                k1 = kpts[i1]
                u1_nR = to_real_space(k1.psit)
                rsk1 = RSKPoint(u1_nR, k1.proj.broadcast(),
                                k1.f_n, k1.k_c,
                                k1.weight, k1.dPdR_aniv)
                lasti1 = i1
            if i2 == i1:
                if s == self.s0:
                    rsk2 = rsk1
                else:
                    N = len(rsk1.u_nR)
                    S = comm.size
                    B = (N + S - 1) // S
                    na = min(B * comm.rank, N)
                    nb = min(na + B, N)
                    rsk2 = RSKPoint(rsk1.u_nR[na:nb],
                                    rsk1.proj.view(na, nb),
                                    rsk1.f_n[na:nb],
                                    rsk1.k_c,
                                    rsk1.weight)
                lasti2 = i2
            elif i2 != lasti2:
                k2 = kpts[i2]
                N = len(k2.psit.array)
                S = comm.size
                B = (N + S - 1) // S
                na = min(B * comm.rank, N)
                nb = min(na + B, N)
                u2_nR = to_real_space(k2.psit, na, nb)
                rsk2 = RSKPoint(u2_nR, k2.proj.broadcast().view(na, nb),
                                k2.f_n[na:nb], k2.k_c,
                                k2.weight)
                lasti2 = i2

            yield (i1, i2, s, rsk1,
                   self.apply_symmetry(s, rsk2, wfs, spos_ac),
                   count)
