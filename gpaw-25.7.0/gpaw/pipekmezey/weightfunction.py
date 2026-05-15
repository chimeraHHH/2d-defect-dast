r""" Class which builds a weight function for each atom
    in a molecule using simple atom centred gaussians.

    This is the Hirshfeld scheme:
               A
     A        p (r)
    w (r) = ---------
             mol
             \   n
             /  p (r)
             --
             n

"""

import numpy as np
from math import pi
from ase.units import Bohr

# Cut-offs: [Ang]
Rc = {'Fe': 3.762,
      'O': 3.762,
      # 'H' : 3.762
      }

# Gauss-width: [Ang]
mu = {'Fe': 1.0,
      'O': 0.6,
      # 'H' : 0.6
      }


class WeightFunc:

    def __init__(self, gd, atoms, indexes, Rc=Rc, mu=mu):
        """ Given a grid-descriptor, atoms object and an index list
            construct a weight function defined by:

        """

        self.gd = gd  # Grid descriptor
        self.atoms = atoms
        self.indexes = indexes

        # Construct Rc dict
        new = {}
        for a in self.atoms:
            if a.symbol in Rc:
                new[a.symbol] = Rc[a.symbol]
            else:
                new[a.symbol] = 3.762

        self.Rc = new

        # Construct mu dict
        new_mu = {}
        for a in self.atoms:
            if mu:
                if a.symbol in mu:
                    new_mu[a.symbol] = mu[a.symbol]
            else:
                new_mu[a.symbol] = 0.85

        # Larger atoms may need a bit more width?
        self.mu = new_mu

    def truncated_gaussian(self, dis, mu, Rc):
        # Given mu and Rc construct Gaussian.
        # Gauss. is truncated at Rc

        check = abs(dis) <= Rc / Bohr
        # Make gaussian
        gauss = 1.0 / (mu * np.sqrt(2.0 * pi)) * \
            np.exp(- dis ** 2 / (2.0 * mu))
        # Apply cut-off and send
        return (gauss * check)

    def get_distance_vectors(self, pos):
        # Given atom position [Bohr], grab distances to all
        # grid-points (gpts) - employ MIC where appropriate.

        # Scaled positions of gpts on some cpu, relative to all
        s_G = (np.indices(self.gd.n_c, float).T +
               self.gd.beg_c) / self.gd.N_c
        # print self.gd.n_c, self.gd.beg_c
        # Subtract scaled distance from atom to box boundary
        s_G -= np.linalg.solve(self.gd.cell_cv.T, pos)
        # MIC
        s_G -= self.gd.pbc_c * (2 * s_G).astype(int)
        # Apparently doing this twice works better...
        s_G -= self.gd.pbc_c * (2 * s_G).astype(int)
        # x,y,z distances
        xyz = np.dot(s_G, self.gd.cell_cv).T.copy()
        #
        return np.sqrt((xyz ** 2).sum(axis=0))

    def construct_total_density(self, atoms):
        # Add to empty grid
        empty = self.gd.zeros()

        for atom in atoms:
            charge = atom.number
            symbol = atom.symbol

            pos = atom.position / Bohr

            dis = self.get_distance_vectors(pos)

            empty += charge * self.truncated_gaussian(
                dis, self.mu[symbol], self.Rc[symbol])
        #
        return empty

    def construct_weight_function(self):
        # Grab atomic / molecular density
        dens_n = self.construct_total_density(
            self.atoms[self.indexes])
        # Grab total density
        dens = self.construct_total_density(self.atoms)
        # Check zero elements
        check = dens == 0
        # Add constant to zeros ...
        dens += check * 1.0
        # make and send
        return dens_n / dens


class WignerSeitz:
    """ Construct weight function based on Wigner-Seitz

               | 1 if (r-R) < (r-R')
        w(r) = |
               | 0 if (r-R) > (r-R')

        for atom A at pos R

    """

    def __init__(self, gd, atoms, index):
        #
        self.gd = gd
        self.atoms = atoms
        self.index = index

    def get_distance_vectors(self, pos):
        #
        # Given atom position [Bohr], grab distances to all
        # grid-points (gpts) - employ MIC where appropriate.

        # Scaled positions of gpts on some cpu, relative to all
        s_G = (np.indices(self.gd.n_c, float).T +
               self.gd.beg_c) / self.gd.N_c
        #
        # Subtract scaled distance from atom to box boundary
        s_G -= np.linalg.solve(self.gd.cell_cv.T, pos)
        # MIC
        s_G -= self.gd.pbc_c * (2 * s_G).astype(int)
        # Apparently doing this twice works better...
        s_G -= self.gd.pbc_c * (2 * s_G).astype(int)
        # x,y,z distances
        xyz = np.dot(s_G, self.gd.cell_cv).T.copy()
        #
        return np.sqrt((xyz ** 2).sum(axis=0))

    def construct_weight_function(self):
        # Grab distances of A to all gpts
        pos = self.atoms[self.index].position / Bohr
        dis_n = self.get_distance_vectors(pos)
        #
        empty = self.gd.zeros()
        norm = self.gd.zeros()
        #
        empty += 1
        norm += 1
        #
        for i, atom in enumerate(self.atoms):
            #
            if i == self.index:
                continue
            else:
                #
                pos = atom.position / Bohr
                dis_b = self.get_distance_vectors(pos)
                #
                check = dis_n <= dis_b  #
                n_c = dis_n == dis_b  #
                # 0 if longer, 1 if shorter, 1/no.(Ra=Ra') if same
                empty *= check
                norm += n_c * 1.0

        return empty / norm
