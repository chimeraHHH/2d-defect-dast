from math import pi
from gpaw.core.plane_waves import PWDesc
from gpaw.core.pwacf import PWAtomCenteredFunctions


class SpiralPWDesc:
    def __init__(self,
                 pw: PWDesc,
                 qspiral_v):
        self.pw = pw
        self.qspiral_v = qspiral_v
        self.dtype = complex
        self.maxmysize = pw.maxmysize
        self.comm = pw.comm
        self.myshape = pw.myshape
        self.G_plus_k_Gv = pw.G_plus_k_Gv + qspiral_v
        self.ekin_G = 0.5 * (self.G_plus_k_Gv**2).sum(1)
        self.kpt = pw.kpt_c + pw.cell_cv @ qspiral_v / (2 * pi)
        self.kpt_c = self.kpt
        self.cell = pw.cell
        self.dv = pw.dv


class SpiralPWACF:
    def __init__(self, functions, positions, pw,
                 atomdist,
                 qspiral_v):
        self.pt_saiG = [
            PWAtomCenteredFunctions(
                functions,
                positions,
                SpiralPWDesc(pw, 0.5 * sign * qspiral_v),
                atomdist=atomdist)
            for sign in [1, -1]]

    def empty(self, dims, comm):
        return self.pt_saiG[0].empty(dims, comm)

    def integrate(self, psit_nsG, out):
        P_ansi = out
        for s, pt_aiG in enumerate(self.pt_saiG):
            pt_aiG._lazy_init()
            pt_aiG._lfc.integrate(psit_nsG.data[:, s],
                                  {a: P_nsi[:, s]
                                   for a, P_nsi in P_ansi.items()})

    def add_to(self, r_nsG, P_ansi):
        for s, pt_aiG in enumerate(self.pt_saiG):
            # pt_aiG._lazy_init()
            pt_aiG._lfc.add(r_nsG.data[:, s],
                            {a: P_nsi[:, s]
                             for a, P_nsi in P_ansi.items()})
