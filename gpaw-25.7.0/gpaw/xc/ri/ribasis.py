from dataclasses import replace
from gpaw.basis_data import BasisFunction
from collections import defaultdict


def generate_ri_basis(basis, accuracy):
    lmax = 2

    # TODO: Hartree
    def poisson(n_g, l):
        return Hartree(basis.rgd, n_g, l)

    # Auxiliary basis functions per angular momentum channel
    auxt_lng = defaultdict(lambda: [])

    ribf_j = []

    def add(aux_g, l, rc=None):
        ribf = BasisFunction(n=None, l=l, rc=rc, phit_g=aux_g,
                             type='auxiliary')
        ribf_j.append(ribf)

    def basisloop():
        for j, bf in enumerate(basis.bf_j):
            yield j, bf.l, bf.rc, bf.phit_g

    # Double basis function loop to create product orbitals
    for j1, l1, rc1, phit1_g in basisloop():
        for j2, l2, rc2, phit2_g in basisloop():
            # Loop only over ordered pairs
            if j1 > j2:
                continue

            # Loop over all possible angular momentum states what the
            # product l1 x l2 creates.
            for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                if l > lmax:
                    continue

                # min: The support of basis function is the intersection
                # of the individual supports.
                add(phit1_g * phit2_g, l, rc=min(rc1, rc2))

    for l, auxt_ng in auxt_lng.items():
        print(l, auxt_ng)
        print(f'    l={l}')
        for n, auxt_g in enumerate(auxt_ng):
            print(f'        {n}')

    return replace(basis, ribf_j=ribf_j)


def Hartree(rgd, n_g, l):
    v_g = rgd.poisson(n_g, l)
    v_g[1:] /= rgd.r_g[1:]
    v_g[0] = v_g[1]
    return v_g
