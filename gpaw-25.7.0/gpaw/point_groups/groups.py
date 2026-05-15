from math import pi
import numpy as np
from scipy.spatial.transform import Rotation
"""
The point groups are written as classes that inherit the class
Pointgroup and contain the information about
- the character table in irreducible form
- representations that correspond to translations in x, y and z
- whether the point group contains complex representations

About complex point groups:
The classes of complex groups (such as Th) are tagged
as that they contain the attribute `complex`. For those groups,
only the real character table is given which is used as such
in the reduction formula. For the representations of these
character tables, the regular orthogonality rules between the
rows and columns do not apply due to the lost information about
the imaginary parts.

"""


def rotation(vec):
    rot = Rotation.from_rotvec(vec)
    try:
        return rot.as_matrix()  # new in scipy-1.4.0
    except AttributeError:
        return rot.as_dcm()  # will be removed in scipy-1.6.0


class Pointgroup:
    def unit(self, data):
        return np.eye(3)

    def rotate_mainaxis(self, angle, data=None):
        return rotation([0, 0, angle / 180 * pi])

    def rotate(self, angle, data=None, axis='z'):
        """
        Parameters
        ----------
        data : array_like
            3D data (a wave function)
        angle : float
            Rotational angle in degrees
        axis : string, optional
            The Rotational axis ('x', 'y' or 'z')
        """
        vec = [0, 0, 0]
        vec['xyz'.index(axis)] = angle / 180 * pi
        return rotation(vec).dot(data)

    def C2prime(self, data, angle):
        """180 degree rotation around the secondary axis"""
        newdata = self.rotate(angle=angle, data=data, axis='z')
        newdata2 = self.rotate(angle=180., data=newdata, axis='x')
        return self.rotate(angle=-angle, data=newdata2,
                           axis='z')  # rotate back

    def S(self, angle, data):
        """
        Rotate and mirror
        """
        rotated = self.rotate(angle=angle, data=data, axis='z')
        return self.mirror_xy(rotated)

    def invert(self, data):
        return self.mirror_xy(self.mirror_yz(self.mirror_xz(data)))

    def mirror_xy(self, data):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1.0]]).dot(data)

    def mirror_xz(self, data):
        return np.array([[1, 0, 0], [0, -1.0, 0], [0, 0, 1]]).dot(data)

    def mirror_yz(self, data):
        return np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).dot(data)


class D5(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit),
                           ('1C5', self.rotate_mainaxis(angle=72.)),
                           ('4C5', self.rotate_mainaxis(angle=-72.)),
                           ('2C5', self.rotate_mainaxis(angle=2 * 72.)),
                           ('3C5', self.rotate_mainaxis(angle=-2 * 72.)),
                           ('C2_0', self.C2(angle=0 * 72.)),
                           ('C2_1', self.C2(angle=1 * 72.)),
                           ('C2_2', self.C2(angle=2 * 72.)),
                           ('C2_3', self.C2(angle=3 * 72.)),
                           ('C2_4', self.C2(angle=4 * 72.))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['A1', 'A2', 'E1', 'E2']
        self.character_table = [
            [1., 1., 1., 1.], [1., 1., 1., -1.],
            [2., 2 * np.cos(2 * np.pi / 5.), 2 * np.cos(4 * np.pi / 5.), 0.],
            [2., 2 * np.cos(4 * np.pi / 5.), 2 * np.cos(2 * np.pi / 5.), 0.]
        ]
        self.nof_operations = [1, 2, 2, 5]
        self.Tx_i = 2  # row index for representation of translation vector x
        self.Ty_i = 2
        self.Tz_i = 1

    def __str__(self):
        return 'D5'

    def C2(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.C2prime(data, angle)

        if data is None:
            return do_it
        return do_it(data)


class D5h(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit),
                           ('1C5', self.rotate_mainaxis(angle=72.)),
                           ('4C5', self.rotate_mainaxis(angle=-72.)),
                           ('2C5', self.rotate_mainaxis(angle=2 * 72.)),
                           ('3C5', self.rotate_mainaxis(angle=-2 * 72.)),
                           ('C2_0', self.C2(angle=0 * 72.)),
                           ('C2_1', self.C2(angle=1 * 72.)),
                           ('C2_2', self.C2(angle=2 * 72.)),
                           ('C2_3', self.C2(angle=3 * 72.)),
                           ('C2_4', self.C2(angle=4 * 72.)),
                           ('sigma_h', self.sigma_h),
                           ('S5_1', self.S5(angle=72.)),
                           ('S5_2', self.S5(angle=-72.)),
                           ('S53_1', self.S53(angle=72.)),
                           ('S53_2', self.S53(angle=-72.)),
                           ('sigma_v0', self.sigma_v(angle=0.)),
                           ('sigma_v1', self.sigma_v(angle=72.)),
                           ('sigma_v4', self.sigma_v(angle=-72.)),
                           ('sigma_v2', self.sigma_v(angle=2 * 72.)),
                           ('sigma_v3', self.sigma_v(angle=-2 * 72.))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = [
            'A\'1', 'A\'2', 'E\'1', 'E\'2', 'A\'\'1', 'A\'\'2', 'E\'\'1',
            'E\'\'2'
        ]
        self.character_table = [
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., -1., 1., 1., 1., -1.],
            [2., 2 * np.cos(2 * np.pi / 5.),
             2 * np.cos(4 * np.pi / 5.), 0., 2.,
             2 * np.cos(2 * np.pi / 5.),
             2 * np.cos(4 * np.pi / 5.), 0.],
            [2.,
             2 * np.cos(4 * np.pi / 5.),
             2 * np.cos(2 * np.pi / 5.),
             0.,
             2.,
             2 * np.cos(4 * np.pi / 5.),
             2 * np.cos(2 * np.pi / 5.),
             0.],
            [1., 1., 1., 1., -1., -1., -1., -1.],
            [1., 1., 1., -1., -1., -1., -1., 1.],
            [2., 2 * np.cos(2 * np.pi / 5.),
             2 * np.cos(4 * np.pi / 5.), 0., -2.,
             -2 * np.cos(2 * np.pi / 5.),
             -2 * np.cos(4 * np.pi / 5.), 0.],
            [2.,
             2 * np.cos(4 * np.pi / 5.),
             2 * np.cos(2 * np.pi / 5.),
             0.,
             -2.,
             -2 * np.cos(4 * np.pi / 5.),
             -2 * np.cos(2 * np.pi / 5.),
             0.]]
        self.nof_operations = [1, 2, 2, 5, 1, 2, 2, 5]
        self.Tx_i = 2
        self.Ty_i = 2
        self.Tz_i = 5

    def __str__(self):
        return 'D5h'

    def C2(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.C2prime(data, angle)

        if data is None:
            return do_it
        return do_it(data)

    def sigma_h(self, data=None):
        return self.mirror_xy(data)

    def S5(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.S(angle=angle, data=data)

        if data is None:
            return do_it
        return do_it(data)

    def S53(self, angle, data=None):
        angle = angle

        def do_it(data):
            data = self.S(angle=angle, data=data)
            data = self.S(angle=angle, data=data)
            data = self.S(angle=angle, data=data)
            return data

        if data is None:
            return do_it
        return do_it(data)

    def sigma_v(self, angle, data=None):
        # first rotate so that the plane is xz plane, flip, and rotate back
        angle = angle

        def do_it(data):
            first = self.rotate(data=data, angle=angle)
            second = self.mirror_xz(first)
            third = self.rotate(data=second, angle=-angle)
            return third

        if data is None:
            return do_it
        return do_it(data)


class Ih(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        # some geometric math:
        a = 1.
        r_cap = np.sin(2 * np.pi / 5) * a
        r_side = np.sqrt(3) * (3 + np.sqrt(5)) / 12. * a
        r_edge = np.cos(np.pi / 5) * a
        r_captoside = 1 / np.sqrt(3) * a
        r_captoedge = 0.5 * a
        r_height = np.sqrt(3) / 2. * a
        self.angle_captocap = np.arccos(((r_cap**2 + r_cap**2) - a**2) /
                                        (2 * r_cap * r_cap)) * 180. / np.pi
        self.angle_captoside = np.arccos(
            ((r_side**2 + r_cap**2) - r_captoside**2) /
            (2 * r_side * r_cap)) * 180. / np.pi
        self.angle_captoedge = np.arccos(
            ((r_edge**2 + r_cap**2) - r_captoedge**2) /
            (2 * r_edge * r_cap)) * 180. / np.pi
        self.angle_captoedge2 = np.arccos(
            ((r_edge**2 + r_cap**2) - r_height**2) /
            (2 * r_edge * r_cap)) * 180. / np.pi
        self.angle_sidetoside = 180. - (2 * self.angle_captoside +
                                        self.angle_captocap)

        self.operations = [
            ('E', self.unit), ('01C5', self.rotate_mainaxis(angle=72.)),
            ('02C5', self.rotate_mainaxis(angle=-72.)),
            ('03C5', self.rotate_othermainaxis(numberofcap=0, angle=72.)),
            ('04C5', self.rotate_othermainaxis(numberofcap=0, angle=-72.)),
            ('05C5', self.rotate_othermainaxis(numberofcap=1, angle=72.)),
            ('06C5', self.rotate_othermainaxis(numberofcap=1, angle=-72.)),
            ('07C5', self.rotate_othermainaxis(numberofcap=2, angle=72.)),
            ('08C5', self.rotate_othermainaxis(numberofcap=2, angle=-72.)),
            ('09C5', self.rotate_othermainaxis(numberofcap=3, angle=72.)),
            ('10C5', self.rotate_othermainaxis(numberofcap=3, angle=-72.)),
            ('11C5', self.rotate_othermainaxis(numberofcap=4, angle=72.)),
            ('12C5', self.rotate_othermainaxis(numberofcap=4, angle=-72.)),
            ('01C52', self.rotate_mainaxis(angle=2 * 72.)),
            ('02C52', self.rotate_mainaxis(angle=-2 * 72.)),
            ('03C52', self.rotate_othermainaxis(numberofcap=0, angle=2 * 72.)),
            ('04C52', self.rotate_othermainaxis(numberofcap=0,
                                                angle=-2 * 72.)),
            ('05C52', self.rotate_othermainaxis(numberofcap=1, angle=2 * 72.)),
            ('06C52', self.rotate_othermainaxis(numberofcap=1,
                                                angle=-2 * 72.)),
            ('07C52', self.rotate_othermainaxis(numberofcap=2, angle=2 * 72.)),
            ('08C52', self.rotate_othermainaxis(numberofcap=2,
                                                angle=-2 * 72.)),
            ('09C52', self.rotate_othermainaxis(numberofcap=3, angle=2 * 72.)),
            ('10C52', self.rotate_othermainaxis(numberofcap=3,
                                                angle=-2 * 72.)),
            ('11C52', self.rotate_othermainaxis(numberofcap=4, angle=2 * 72.)),
            ('12C52', self.rotate_othermainaxis(numberofcap=4,
                                                angle=-2 * 72.)),
            ('01C3', self.rotate_C3(numberofside=0, angle=120.)),
            ('02C3', self.rotate_C3(numberofside=0, angle=-120.)),
            ('03C3', self.rotate_C3(numberofside=1, angle=120.)),
            ('04C3', self.rotate_C3(numberofside=1, angle=-120.)),
            ('05C3', self.rotate_C3(numberofside=2, angle=120.)),
            ('06C3', self.rotate_C3(numberofside=2, angle=-120.)),
            ('07C3', self.rotate_C3(numberofside=3, angle=120.)),
            ('08C3', self.rotate_C3(numberofside=3, angle=-120.)),
            ('09C3', self.rotate_C3(numberofside=4, angle=120.)),
            ('10C3', self.rotate_C3(numberofside=4, angle=-120.)),
            ('11C3', self.rotate_C3(numberofside=5, angle=120.)),
            ('12C3', self.rotate_C3(numberofside=5, angle=-120.)),
            ('13C3', self.rotate_C3(numberofside=6, angle=120.)),
            ('14C3', self.rotate_C3(numberofside=6, angle=-120.)),
            ('15C3', self.rotate_C3(numberofside=7, angle=120.)),
            ('16C3', self.rotate_C3(numberofside=7, angle=-120.)),
            ('17C3', self.rotate_C3(numberofside=8, angle=120.)),
            ('18C3', self.rotate_C3(numberofside=8, angle=-120.)),
            ('19C3', self.rotate_C3(numberofside=9, angle=120.)),
            ('20C3', self.rotate_C3(numberofside=9, angle=-120.)),
            ('01C2', self.C2(numberofedge=0)), ('02C2',
                                                self.C2(numberofedge=1)),
            ('03C2', self.C2(numberofedge=2)), ('04C2',
                                                self.C2(numberofedge=3)),
            ('05C2', self.C2(numberofedge=4)), ('06C2',
                                                self.C2(numberofedge=5)),
            ('07C2', self.C2(numberofedge=6)), ('08C2',
                                                self.C2(numberofedge=7)),
            ('09C2', self.C2(numberofedge=8)),
            ('10C2', self.C2(numberofedge=9)),
            ('11C2', self.C2(numberofedge=10)),
            ('12C2', self.C2(numberofedge=11)),
            ('13C2', self.C2(numberofedge=12)),
            ('14C2', self.C2(numberofedge=13)),
            ('15C2', self.C2(numberofedge=14)), ('i', self.inversion),
            ('01S10', self.S10(angle=36.)), ('02S10', self.S10(angle=-36.)),
            ('03S10', self.S10_othercap(numberofcap=0, angle=36.)),
            ('04S10', self.S10_othercap(numberofcap=0, angle=-36.)),
            ('05S10', self.S10_othercap(numberofcap=1, angle=36.)),
            ('06S10', self.S10_othercap(numberofcap=1, angle=-36.)),
            ('07S10', self.S10_othercap(numberofcap=2, angle=36.)),
            ('08S10', self.S10_othercap(numberofcap=2, angle=-36.)),
            ('09S10', self.S10_othercap(numberofcap=3, angle=36.)),
            ('10S10', self.S10_othercap(numberofcap=3, angle=-36.)),
            ('11S10', self.S10_othercap(numberofcap=4, angle=36.)),
            ('12S10', self.S10_othercap(numberofcap=4, angle=-36.)),
            ('01S10_3', self.S10_3(angle=36.)),
            ('02S10_3', self.S10_3(angle=-36.)),
            ('03S10_3', self.S10_3_othercap(numberofcap=0, angle=36.)),
            ('04S10_3', self.S10_3_othercap(numberofcap=0, angle=-36.)),
            ('05S10_3', self.S10_3_othercap(numberofcap=1, angle=36.)),
            ('06S10_3', self.S10_3_othercap(numberofcap=1, angle=-36.)),
            ('07S10_3', self.S10_3_othercap(numberofcap=2, angle=36.)),
            ('08S10_3', self.S10_3_othercap(numberofcap=2, angle=-36.)),
            ('09S10_3', self.S10_3_othercap(numberofcap=3, angle=36.)),
            ('10S10_3', self.S10_3_othercap(numberofcap=3, angle=-36.)),
            ('11S10_3', self.S10_3_othercap(numberofcap=4, angle=36.)),
            ('12S10_3', self.S10_3_othercap(numberofcap=4, angle=-36.)),
            ('01S6', self.S6(numberofside=0, angle=60.)),
            ('02S6', self.S6(numberofside=0, angle=-60.)),
            ('03S6', self.S6(numberofside=1, angle=60.)),
            ('04S6', self.S6(numberofside=1, angle=-60.)),
            ('05S6', self.S6(numberofside=2, angle=60.)),
            ('06S6', self.S6(numberofside=2, angle=-60.)),
            ('07S6', self.S6(numberofside=3, angle=60.)),
            ('08S6', self.S6(numberofside=3, angle=-60.)),
            ('09S6', self.S6(numberofside=4, angle=60.)),
            ('10S6', self.S6(numberofside=4, angle=-60.)),
            ('11S6', self.S6(numberofside=5, angle=60.)),
            ('12S6', self.S6(numberofside=5, angle=-60.)),
            ('13S6', self.S6(numberofside=6, angle=60.)),
            ('14S6', self.S6(numberofside=6, angle=-60.)),
            ('15S6', self.S6(numberofside=7, angle=60.)),
            ('16S6', self.S6(numberofside=7, angle=-60.)),
            ('17S6', self.S6(numberofside=8, angle=60.)),
            ('18S6', self.S6(numberofside=8, angle=-60.)),
            ('19S6', self.S6(numberofside=9, angle=60.)),
            ('20S6', self.S6(numberofside=9, angle=-60.)),
            ('01sigma', self.sigma(numberofedge=0)),
            ('02sigma', self.sigma(numberofedge=1)),
            ('03sigma', self.sigma(numberofedge=2)),
            ('04sigma', self.sigma(numberofedge=3)),
            ('05sigma', self.sigma(numberofedge=4)),
            ('06sigma', self.sigma(numberofedge=5)),
            ('07sigma', self.sigma(numberofedge=6)),
            ('08sigma', self.sigma(numberofedge=7)),
            ('09sigma', self.sigma(numberofedge=8)),
            ('10sigma', self.sigma(numberofedge=9)),
            ('11sigma', self.sigma(numberofedge=10)),
            ('12sigma', self.sigma(numberofedge=11)),
            ('13sigma', self.sigma(numberofedge=12)),
            ('14sigma', self.sigma(numberofedge=13)),
            ('15sigma', self.sigma(numberofedge=14))
        ]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = [
            'Ag', 'T1g', 'T2g', 'Gg', 'Hg', 'Au', 'T1u', 'T2u', 'Gu', 'Hu'
        ]
        self.character_table = [
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [
                3., -2 * np.cos(4 * np.pi / 5.), -2 * np.cos(2 * np.pi / 5.),
                0., -1., 3., -2 * np.cos(2 * np.pi / 5.),
                -2 * np.cos(4 * np.pi / 5.), 0., -1.
            ],
            [
                3., -2 * np.cos(2 * np.pi / 5.), -2 * np.cos(4 * np.pi / 5.),
                0., -1., 3., -2 * np.cos(4 * np.pi / 5.),
                -2 * np.cos(2 * np.pi / 5.), 0., -1.
            ], [4., -1., -1., 1., 0., 4., -1., -1., 1., 0.],
            [5., 0., 0., -1., 1., 5., 0., 0., -1., 1.],
            [1., 1., 1., 1., 1., -1., -1., -1., -1., -1.],
            [
                3., -2 * np.cos(4 * np.pi / 5.), -2 * np.cos(2 * np.pi / 5.),
                0., -1., -3., 2 * np.cos(2 * np.pi / 5.),
                2 * np.cos(4 * np.pi / 5.), 0., 1.
            ],
            [
                3., -2 * np.cos(2 * np.pi / 5.), -2 * np.cos(4 * np.pi / 5.),
                0., -1., -3., 2 * np.cos(4 * np.pi / 5.),
                2 * np.cos(2 * np.pi / 5.), 0., 1.
            ], [4., -1., -1., 1., 0., -4., 1., 1., -1., 0.],
            [5., 0., 0., -1., 1., -5., 0., 0., 1., -1.]
        ]
        self.nof_operations = [1, 12, 12, 20, 15, 1, 12, 12, 20, 15]
        self.Tx_i = 6
        self.Ty_i = 6
        self.Tz_i = 6

    def __str__(self):
        return 'Ih'

    def rotate_othermainaxis(self, numberofcap, angle, data=None):
        angle = angle
        numberofcap = numberofcap

        def do_it(data):
            # first, bring another cap to z-axis:
            data = self.rotate(angle=numberofcap * 72., data=data, axis='z')
            data = self.rotate(angle=self.angle_captocap, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-self.angle_captocap, data=data, axis='y')
            data = self.rotate(angle=-numberofcap * 72., data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def rotate_C3(self, numberofside, angle, data=None):
        angle = angle
        numberofside = numberofside
        n_is_even = (numberofside % 2 == 0)
        # first, bring a face center to z-axis:
        angle1 = int((numberofside + 1e-4) / 2.) * 72.
        if n_is_even:
            angle2 = -(self.angle_captoside + self.angle_sidetoside)
        else:
            angle2 = -self.angle_captoside

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=angle2, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def C2(self, numberofedge, angle=180., data=None):
        angle = angle
        numberofedge = numberofedge
        n_is_g1 = (numberofedge % 3 == 0)
        n_is_g2 = (numberofedge % 3 == 1)
        # first, bring an edge center to z-axis:
        angle1 = int((numberofedge + 1e-4) / 3.) * 72. + 36.
        if n_is_g1:
            angle2 = 0.
            angle3 = -self.angle_captoedge
        elif n_is_g2:
            angle2 = 36.
            angle3 = -self.angle_captoedge2
        else:
            angle2 = 18.
            angle3 = -90.

        def do_it(data):
            data = self.rotate(angle=angle1 + angle2, data=data, axis='z')
            data = self.rotate(angle=angle3, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-angle3, data=data, axis='y')
            data = self.rotate(angle=-(angle1 + angle2), data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def inversion(self, data=None):
        return self.invert(data)

    def S10(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.S(data=data, angle=angle)

        if data is None:
            return do_it
        return do_it(data)

    def S10_othercap(self, numberofcap, angle, data=None):
        angle = angle
        numberofcap = numberofcap

        def do_it(data):
            # first, bring another cap to z-axis:
            data = self.rotate(angle=numberofcap * 72., data=data, axis='z')
            data = self.rotate(angle=self.angle_captocap, data=data, axis='y')

            # do the actual S-operation:
            data = self.S(angle=angle, data=data)

            # rotate back:
            data = self.rotate(angle=-self.angle_captocap, data=data, axis='y')
            data = self.rotate(angle=-numberofcap * 72., data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def S10_3(self, angle, data=None):
        angle = angle

        def do_it(data):
            first = self.S(data=data, angle=angle)
            second = self.S(data=first, angle=angle)
            third = self.S(data=second, angle=angle)
            return third

        if data is None:
            return do_it
        return do_it(data)

    def S10_3_othercap(self, numberofcap, angle, data=None):
        angle = angle
        numberofcap = numberofcap

        def do_it(data):
            # first, bring another cap to z-axis:
            data = self.rotate(angle=numberofcap * 72., data=data, axis='z')
            data = self.rotate(angle=self.angle_captocap, data=data, axis='y')

            # do the actual S-operation:
            data = self.S(data=data, angle=angle)
            data = self.S(data=data, angle=angle)
            data = self.S(data=data, angle=angle)

            # rotate back:
            data = self.rotate(angle=-self.angle_captocap, data=data, axis='y')
            data = self.rotate(angle=-numberofcap * 72., data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def S6(self, numberofside, angle, data=None):
        angle = angle
        numberofside = numberofside
        n_is_even = (numberofside % 2 == 0)
        # first, bring a face center to z-axis:
        angle1 = int((numberofside + 1e-4) / 2.) * 72.
        if n_is_even:
            angle2 = (self.angle_captoside + self.angle_sidetoside)
        else:
            angle2 = self.angle_captoside

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=-angle2, data=data, axis='y')

            # do the actual rotation:
            data = self.S(angle=angle, data=data)

            # rotate back:
            data = self.rotate(angle=angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def sigma(self, numberofedge, data=None):
        numberofedge = numberofedge
        n_is_g1 = (numberofedge % 3 == 0)
        n_is_g2 = (numberofedge % 3 == 1)
        # first, bring an edge center to z-axis:
        angle1 = int((numberofedge + 1e-4) / 3.) * 72. + 36.
        if n_is_g1:
            angle2 = 0.
            angle3 = self.angle_captoedge
            angle4 = 0.
        elif n_is_g2:
            angle2 = 36.
            angle3 = self.angle_captoedge2
            angle4 = 90.
        else:
            angle2 = 18.
            angle3 = 90.
            angle4 = self.angle_captoedge

        def do_it(data):
            data = self.rotate(angle=angle1 + angle2, data=data, axis='z')
            data = self.rotate(angle=-angle3, data=data, axis='y')
            data = self.rotate(angle=angle4, data=data, axis='z')

            # do the actual rotation:
            data = self.mirror_xz(data)

            # rotate back:
            data = self.rotate(angle=-angle4, data=data, axis='z')
            data = self.rotate(angle=angle3, data=data, axis='y')
            data = self.rotate(angle=-(angle1 + angle2), data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)


class Ico(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        # some geometric math:
        a = 1.
        r_cap = np.sin(2 * np.pi / 5) * a
        r_side = np.sqrt(3) * (3 + np.sqrt(5)) / 12. * a
        r_edge = np.cos(np.pi / 5) * a
        r_captoside = 1 / np.sqrt(3) * a
        r_captoedge = 0.5 * a
        r_height = np.sqrt(3) / 2. * a
        self.angle_captocap = np.arccos(((r_cap**2 + r_cap**2) - a**2) /
                                        (2 * r_cap * r_cap)) * 180. / np.pi
        self.angle_captoside = np.arccos(
            ((r_side**2 + r_cap**2) - r_captoside**2) /
            (2 * r_side * r_cap)) * 180. / np.pi
        self.angle_captoedge = np.arccos(
            ((r_edge**2 + r_cap**2) - r_captoedge**2) /
            (2 * r_edge * r_cap)) * 180. / np.pi
        self.angle_captoedge2 = np.arccos(
            ((r_edge**2 + r_cap**2) - r_height**2) /
            (2 * r_edge * r_cap)) * 180. / np.pi
        self.angle_sidetoside = 180. - (2 * self.angle_captoside +
                                        self.angle_captocap)

        self.operations = [
            ('E', self.unit), ('01C5', self.rotate_mainaxis(angle=72.)),
            ('02C5', self.rotate_mainaxis(angle=-72.)),
            ('03C5', self.rotate_othermainaxis(numberofcap=0, angle=72.)),
            ('04C5', self.rotate_othermainaxis(numberofcap=0, angle=-72.)),
            ('05C5', self.rotate_othermainaxis(numberofcap=1, angle=72.)),
            ('06C5', self.rotate_othermainaxis(numberofcap=1, angle=-72.)),
            ('07C5', self.rotate_othermainaxis(numberofcap=2, angle=72.)),
            ('08C5', self.rotate_othermainaxis(numberofcap=2, angle=-72.)),
            ('09C5', self.rotate_othermainaxis(numberofcap=3, angle=72.)),
            ('10C5', self.rotate_othermainaxis(numberofcap=3, angle=-72.)),
            ('11C5', self.rotate_othermainaxis(numberofcap=4, angle=72.)),
            ('12C5', self.rotate_othermainaxis(numberofcap=4, angle=-72.)),
            ('01C52', self.rotate_mainaxis(angle=2 * 72.)),
            ('02C52', self.rotate_mainaxis(angle=-2 * 72.)),
            ('03C52', self.rotate_othermainaxis(numberofcap=0, angle=2 * 72.)),
            ('04C52', self.rotate_othermainaxis(numberofcap=0,
                                                angle=-2 * 72.)),
            ('05C52', self.rotate_othermainaxis(numberofcap=1, angle=2 * 72.)),
            ('06C52', self.rotate_othermainaxis(numberofcap=1,
                                                angle=-2 * 72.)),
            ('07C52', self.rotate_othermainaxis(numberofcap=2, angle=2 * 72.)),
            ('08C52', self.rotate_othermainaxis(numberofcap=2,
                                                angle=-2 * 72.)),
            ('09C52', self.rotate_othermainaxis(numberofcap=3, angle=2 * 72.)),
            ('10C52', self.rotate_othermainaxis(numberofcap=3,
                                                angle=-2 * 72.)),
            ('11C52', self.rotate_othermainaxis(numberofcap=4, angle=2 * 72.)),
            ('12C52', self.rotate_othermainaxis(numberofcap=4,
                                                angle=-2 * 72.)),
            ('01C3', self.rotate_C3(numberofside=0, angle=120.)),
            ('02C3', self.rotate_C3(numberofside=0, angle=-120.)),
            ('03C3', self.rotate_C3(numberofside=1, angle=120.)),
            ('04C3', self.rotate_C3(numberofside=1, angle=-120.)),
            ('05C3', self.rotate_C3(numberofside=2, angle=120.)),
            ('06C3', self.rotate_C3(numberofside=2, angle=-120.)),
            ('07C3', self.rotate_C3(numberofside=3, angle=120.)),
            ('08C3', self.rotate_C3(numberofside=3, angle=-120.)),
            ('09C3', self.rotate_C3(numberofside=4, angle=120.)),
            ('10C3', self.rotate_C3(numberofside=4, angle=-120.)),
            ('11C3', self.rotate_C3(numberofside=5, angle=120.)),
            ('12C3', self.rotate_C3(numberofside=5, angle=-120.)),
            ('13C3', self.rotate_C3(numberofside=6, angle=120.)),
            ('14C3', self.rotate_C3(numberofside=6, angle=-120.)),
            ('15C3', self.rotate_C3(numberofside=7, angle=120.)),
            ('16C3', self.rotate_C3(numberofside=7, angle=-120.)),
            ('17C3', self.rotate_C3(numberofside=8, angle=120.)),
            ('18C3', self.rotate_C3(numberofside=8, angle=-120.)),
            ('19C3', self.rotate_C3(numberofside=9, angle=120.)),
            ('20C3', self.rotate_C3(numberofside=9, angle=-120.)),
            ('01C2', self.C2(numberofedge=0)), ('02C2',
                                                self.C2(numberofedge=1)),
            ('03C2', self.C2(numberofedge=2)), ('04C2',
                                                self.C2(numberofedge=3)),
            ('05C2', self.C2(numberofedge=4)), ('06C2',
                                                self.C2(numberofedge=5)),
            ('07C2', self.C2(numberofedge=6)), ('08C2',
                                                self.C2(numberofedge=7)),
            ('09C2', self.C2(numberofedge=8)),
            ('10C2', self.C2(numberofedge=9)),
            ('11C2', self.C2(numberofedge=10)),
            ('12C2', self.C2(numberofedge=11)),
            ('13C2', self.C2(numberofedge=12)),
            ('14C2', self.C2(numberofedge=13)),
            ('15C2', self.C2(numberofedge=14))
        ]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['A', 'T1', 'T2', 'G', 'H']
        self.character_table = [
            [1., 1., 1., 1., 1.],
            [3., -2 * np.cos(4 * np.pi / 5.),
             -2 * np.cos(2 * np.pi / 5.), 0., -1.],
            [3., -2 * np.cos(2 * np.pi / 5.),
             -2 * np.cos(4 * np.pi / 5.), 0., -1.],
            [4., -1., -1., 1., 0.],
            [5., 0., 0., -1., 1.]]
        self.nof_operations = [1, 12, 12, 20, 15]
        self.Tx_i = 1
        self.Ty_i = 1
        self.Tz_i = 1

    def __str__(self):
        return 'Ico'

    def rotate_othermainaxis(self, numberofcap, angle, data=None):
        angle = angle
        numberofcap = numberofcap

        def do_it(data):
            # first, bring another cap to z-axis:
            data = self.rotate(angle=numberofcap * 72., data=data, axis='z')
            data = self.rotate(angle=self.angle_captocap, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-self.angle_captocap, data=data, axis='y')
            data = self.rotate(angle=-numberofcap * 72., data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def rotate_C3(self, numberofside, angle, data=None):
        angle = angle
        numberofside = numberofside
        n_is_even = (numberofside % 2 == 0)
        # first, bring a face center to z-axis:
        angle1 = int((numberofside + 1e-4) / 2.) * 72.
        if n_is_even:
            angle2 = -(self.angle_captoside + self.angle_sidetoside)
        else:
            angle2 = -self.angle_captoside

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=angle2, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def C2(self, numberofedge, angle=180., data=None):
        angle = angle
        numberofedge = numberofedge
        n_is_g1 = (numberofedge % 3 == 0)
        n_is_g2 = (numberofedge % 3 == 1)
        # first, bring an edge center to z-axis:
        angle1 = int((numberofedge + 1e-4) / 3.) * 72. + 36.
        if n_is_g1:
            angle2 = 0.
            angle3 = -self.angle_captoedge
        elif n_is_g2:
            angle2 = 36.
            angle3 = -self.angle_captoedge2
        else:
            angle2 = 18.
            angle3 = -90.

        def do_it(data):
            data = self.rotate(angle=angle1 + angle2, data=data, axis='z')
            data = self.rotate(angle=angle3, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-angle3, data=data, axis='y')
            data = self.rotate(angle=-(angle1 + angle2), data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)


class Td(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        # some geometric math:
        # a = 1.
        # r_cap = np.sqrt(3/8.) * a
        # r_face = r_cap/3.
        # r_edge = np.sqrt(r_cap * r_face)
        # r_captoside = 1/np.sqrt(3) * a
        # r_captoedge = 0.5*a
        # r_height = np.sqrt(3)/2. * a
        self.angle_captoedge = np.arctan(np.sqrt(2)) * 180. / np.pi
        self.angle_captocap = 2 * self.angle_captoedge
        self.angle_captoface = np.arctan(2. * np.sqrt(2)) * 180. / np.pi

        self.operations = [
            ('E', self.unit), ('1C3', self.rotate_mainaxis(angle=120.)),
            ('2C3', self.rotate_mainaxis(angle=-120.)),
            ('3C3', self.rotate_othermainaxis(numberofcap=0, angle=120.)),
            ('4C3', self.rotate_othermainaxis(numberofcap=0, angle=-120.)),
            ('5C3', self.rotate_othermainaxis(numberofcap=1, angle=120.)),
            ('6C3', self.rotate_othermainaxis(numberofcap=1, angle=-120.)),
            ('7C3', self.rotate_othermainaxis(numberofcap=2, angle=120.)),
            ('8C3', self.rotate_othermainaxis(numberofcap=2, angle=-120.)),
            ('1C2', self.C2(numberofedge=0)), ('2C2', self.C2(numberofedge=1)),
            ('3C2', self.C2(numberofedge=2)),
            ('1S4', self.S4(numberofside=0, angle=90.)),
            ('2S4', self.S4(numberofside=0, angle=-90.)),
            ('3S4', self.S4(numberofside=1, angle=90.)),
            ('4S4', self.S4(numberofside=1, angle=-90.)),
            ('5S4', self.S4(numberofside=2, angle=90.)),
            ('6S4', self.S4(numberofside=2, angle=-90.)),
            ('1sigma', self.sigma(numberofedge=0)),
            ('2sigma', self.sigma(numberofedge=1)),
            ('3sigma', self.sigma(numberofedge=2)),
            ('4sigma', self.sigma(numberofedge=3)),
            ('5sigma', self.sigma(numberofedge=4)),
            ('6sigma', self.sigma(numberofedge=5))
        ]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['A1', 'A2', 'E', 'T1', 'T2']
        self.character_table = [[1., 1., 1., 1., 1.], [1., 1., 1., -1., -1.],
                                [2., -1., 2., 0., 0.], [3., 0., -1., 1., -1.],
                                [3., 0., -1., -1., 1.]]
        self.nof_operations = [1, 8, 3, 6, 6]
        self.Tx_i = 4
        self.Ty_i = 4
        self.Tz_i = 4

    def __str__(self):
        return 'Td'

    def rotate_othermainaxis(self, numberofcap, angle, data=None):
        angle = angle
        numberofcap = numberofcap

        def do_it(data):
            # first, bring another cap to z-axis:
            data = self.rotate(angle=numberofcap * 120., data=data, axis='z')
            data = self.rotate(angle=self.angle_captocap, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-self.angle_captocap, data=data, axis='y')
            data = self.rotate(angle=-numberofcap * 120., data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def C2(self, numberofedge, angle=180., data=None):
        angle = angle
        numberofedge = numberofedge
        # first, bring an edge center to z-axis:
        angle1 = numberofedge * 120.
        angle2 = self.angle_captoedge

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=angle2, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def S4(self, numberofside, angle, data=None):
        angle = angle
        numberofside = numberofside
        # first, bring an edge center to z-axis:
        angle1 = numberofside * 120.
        angle2 = self.angle_captoedge

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=angle2, data=data, axis='y')

            # do the actual rotation:
            data = self.S(angle=angle, data=data)

            # rotate back:
            data = self.rotate(angle=-angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def sigma(self, numberofedge, data=None):
        numberofedge = numberofedge
        # first, bring an edge center to z-axis:
        angle1 = (numberofedge % 3) * 120.
        angle2 = self.angle_captoedge
        angle3 = (numberofedge % 2) * 90.

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=angle2, data=data, axis='y')
            data = self.rotate(angle=angle3, data=data, axis='z')

            # do the actual reflection:
            data = self.mirror_xz(data)

            # rotate back:
            data = self.rotate(angle=-angle3, data=data, axis='z')
            data = self.rotate(angle=-angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)


class Th(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        # some geometric math:
        # a = 1.
        # r_cap = np.sqrt(3/8.) * a
        # r_face = r_cap/3.
        # r_edge = np.sqrt(r_cap * r_face)
        # r_captoside = 1/np.sqrt(3) * a
        # r_captoedge = 0.5*a
        # r_height = np.sqrt(3)/2. * a
        self.angle_captoedge = np.arctan(np.sqrt(2)) * 180. / np.pi
        self.angle_captocap = 2 * self.angle_captoedge
        self.angle_captoface = np.arctan(2. * np.sqrt(2)) * 180. / np.pi

        self.operations = [
            ('E', self.unit), ('1C3', self.rotate_mainaxis(angle=120.)),
            ('2C3', self.rotate_othermainaxis(numberofcap=0, angle=120.)),
            ('3C3', self.rotate_othermainaxis(numberofcap=1, angle=120.)),
            ('4C3', self.rotate_othermainaxis(numberofcap=2, angle=120.)),
            ('1C3_2', self.rotate_mainaxis(angle=-120.)),
            ('2C3_2', self.rotate_othermainaxis(numberofcap=0, angle=-120.)),
            ('3C3_2', self.rotate_othermainaxis(numberofcap=1, angle=-120.)),
            ('4C3_2', self.rotate_othermainaxis(numberofcap=2, angle=-120.)),
            ('1C2', self.C2(numberofedge=0)), ('2C2', self.C2(numberofedge=1)),
            ('3C2', self.C2(numberofedge=2)), ('i', self.inversion),
            ('1S6_5', self.S6(angle=-60.)),
            ('2S6_5', self.S6_othercap(numberofcap=0, angle=-60.)),
            ('3S6_5', self.S6_othercap(numberofcap=1, angle=-60.)),
            ('4S6_5', self.S6_othercap(numberofcap=2, angle=-60.)),
            ('5S6', self.S6(angle=60.)),
            ('6S6', self.S6_othercap(numberofcap=0, angle=60.)),
            ('7S6', self.S6_othercap(numberofcap=1, angle=60.)),
            ('8S6', self.S6_othercap(numberofcap=2, angle=60.)),
            ('1sigma', self.sigma(numberofedge=0)),
            ('2sigma', self.sigma(numberofedge=1)),
            ('3sigma', self.sigma(numberofedge=2))
        ]

        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['Ag', 'Eg', 'Tg', 'Au', 'Eu', 'Tu']
        self.character_table = [
            [1., 1., 1., 1., 1., 1.],
            [2., 2. * np.cos(2 * np.pi / 3.), 2., 2.,
             2. * np.cos(2 * np.pi / 3.), 2.],
            [3., 0., -1., 3., 0., -1.],
            [1., 1., 1., -1., -1., -1.],
            [2., 2. * np.cos(2 * np.pi / 3.), 2., -2.,
             -2. * np.cos(2 * np.pi / 3.), -2.],
            [3., 0., -1., -3., 0., 1.]]
        self.nof_operations = [1, 8, 3, 1, 8, 3]
        self.Tx_i = 5
        self.Ty_i = 5
        self.Tz_i = 5
        self.complex = True

    def __str__(self):
        return 'Th'

    def rotate_othermainaxis(self, numberofcap, angle, data=None):
        angle = angle
        numberofcap = numberofcap

        def do_it(data):
            # first, bring another cap to z-axis:
            data = self.rotate(angle=numberofcap * 120., data=data, axis='z')
            data = self.rotate(angle=self.angle_captocap, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-self.angle_captocap, data=data, axis='y')
            data = self.rotate(angle=-numberofcap * 120., data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def C2(self, numberofedge, angle=180., data=None):
        angle = angle
        numberofedge = numberofedge
        # first, bring an edge center to z-axis:
        angle1 = numberofedge * 120.
        angle2 = self.angle_captoedge

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=angle2, data=data, axis='y')

            # do the actual rotation:
            data = self.rotate(angle=angle, data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def inversion(self, data=None):
        return self.invert(data)

    def S6(self, angle, data=None):
        angle = angle

        def do_it(data):
            data = self.S(angle=angle, data=data)
            return data

        if data is None:
            return do_it
        return do_it(data)

    def S6_othercap(self, numberofcap, angle, data=None):
        angle = angle
        numberofcap = numberofcap

        def do_it(data):
            data = self.rotate(angle=numberofcap * 120., data=data, axis='z')
            data = self.rotate(angle=self.angle_captocap, data=data, axis='y')

            # do the actual rotation:
            data = self.S(angle=angle, data=data)

            # rotate back:
            data = self.rotate(angle=-self.angle_captocap, data=data, axis='y')
            data = self.rotate(angle=-numberofcap * 120, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def sigma(self, numberofedge, data=None):
        numberofedge = numberofedge
        # first, bring an edge center to z-axis:
        angle1 = (numberofedge % 3) * 120.
        angle2 = self.angle_captoedge
        angle3 = 45.

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='z')
            data = self.rotate(angle=angle2, data=data, axis='y')
            data = self.rotate(angle=angle3, data=data, axis='z')

            # do the actual reflection:
            data = self.mirror_xz(data)

            # rotate back:
            data = self.rotate(angle=-angle3, data=data, axis='z')
            data = self.rotate(angle=-angle2, data=data, axis='y')
            data = self.rotate(angle=-angle1, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)


class C2v(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit),
                           ('C2', self.rotate_mainaxis(angle=180.)),
                           ('sigma_v0', self.sigma_v(angle=0.)),
                           ('sigma_v1', self.sigma_v(angle=90.))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['A1', 'A2', 'B1', 'B2']
        self.character_table = [[1., 1., 1., 1.], [1., 1., -1., -1.],
                                [1., -1., 1., -1.], [1., -1., -1., 1.]]
        self.nof_operations = [1, 1, 1, 1]
        self.Tx_i = 2
        self.Ty_i = 3
        self.Tz_i = 0

    def __str__(self):
        return 'C2v'

    def C2(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.C2prime(data, angle)

        if data is None:
            return do_it
        return do_it(data)

    def sigma_v(self, angle, data=None):
        # first rotate so that the plane is xz plane, flip, and rotate back
        angle = angle

        def do_it(data):
            first = self.rotate(data=data, angle=angle)
            second = self.mirror_xz(first)
            third = self.rotate(data=second, angle=-angle)
            return third

        if data is None:
            return do_it
        return do_it(data)


class C3v(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit),
                           ('1C3', self.rotate_mainaxis(angle=120.)),
                           ('2C3', self.rotate_mainaxis(angle=-120.)),
                           ('sigma_v0', self.sigma_v(angle=0.)),
                           ('sigma_v1', self.sigma_v(angle=120.)),
                           ('sigma_v2', self.sigma_v(angle=-120.))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['A1', 'A2', 'E']
        self.character_table = [[1., 1., 1.], [1., 1., -1.], [2., -1., 0.]]
        self.nof_operations = [1, 2, 3]
        self.Tx_i = 2
        self.Ty_i = 2
        self.Tz_i = 0

    def __str__(self):
        return 'C3v'

    def sigma_v(self, angle, data=None):
        # first rotate so that the plane is xz plane, flip, and rotate back
        angle = angle

        def do_it(data):
            first = self.rotate(data=data, angle=angle)
            second = self.mirror_xz(first)
            third = self.rotate(data=second, angle=-angle)
            return third

        if data is None:
            return do_it
        return do_it(data)


class D3h(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit),
                           ('1C3', self.rotate_mainaxis(angle=120.)),
                           ('2C3', self.rotate_mainaxis(angle=-120.)),
                           ('C2_0', self.C2(angle=0 * 120.)),
                           ('C2_1', self.C2(angle=1 * 120.)),
                           ('C2_2', self.C2(angle=2 * 120.)),
                           ('sigma_h', self.sigma_h),
                           ('S3_1', self.S3(angle=120.)),
                           ('S3_2', self.S3(angle=-120.)),
                           ('sigma_v0', self.sigma_v(angle=0.)),
                           ('sigma_v1', self.sigma_v(angle=120.)),
                           ('sigma_v2', self.sigma_v(angle=-120.))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = [
            'A\'1', 'A\'2', 'E\'1', 'A\'\'1', 'A\'\'2', 'E\'\'1'
        ]
        self.character_table = [[1., 1., 1., 1., 1., 1.],
                                [1., 1., -1., 1., 1., -1.],
                                [2., -1., 0., 2., -1., 0.],
                                [1., 1., 1., -1., -1., -1.],
                                [1., 1., -1., -1., -1., 1.],
                                [2., -1., 0., -2., 1., 0.]]
        self.nof_operations = [1, 2, 3, 1, 2, 3]
        self.Tx_i = 2
        self.Ty_i = 2
        self.Tz_i = 4

    def __str__(self):
        return 'D3h'

    def C2(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.C2prime(data, angle)

        if data is None:
            return do_it
        return do_it(data)

    def sigma_h(self, data=None):
        return self.mirror_xy(data)

    def S3(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.S(data=data, angle=angle)

        if data is None:
            return do_it
        return do_it(data)

    def S53(self, angle, data=None):
        angle = angle

        def do_it(data):
            first = self.S(data=data, angle=angle)
            second = self.S(data=first, angle=angle)
            third = self.S(data=second, angle=angle)
            return third

        if data is None:
            return do_it
        return do_it(data)

    def sigma_v(self, angle, data=None):
        # first rotate so that the plane is xz plane, flip, and rotate back
        angle = angle

        def do_it(data):
            first = self.rotate(data=data, angle=angle)
            second = self.mirror_xz(first)
            third = self.rotate(data=second, angle=-angle)
            return third

        if data is None:
            return do_it
        return do_it(data)


class D2d(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit),
                           ('S4_1', self.S4(angle=90.)),
                           ('S4_2', self.S4(angle=-90.)),
                           ('C2_0', self.C2(angle=180.)),
                           ("C2'_0", self.C2p(angle=45.)),
                           ("C2'_1", self.C2p(angle=-45.)),
                           ('sigma_d1', self.sigma_d(angle=0.)),
                           ('sigma_d2', self.sigma_d(angle=90.))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['A\'1', 'A\'2', 'B\'1', 'B\'2', 'E']
        self.character_table = [[1., 1., 1., 1., 1.], [1., 1., 1., -1., -1.],
                                [1., -1., 1., 1., -1.], [1., -1., 1., -1., 1.],
                                [2., 0., -2., 0., 0.]]
        self.nof_operations = [1, 2, 1, 2, 2]
        self.Tx_i = 4
        self.Ty_i = 4
        self.Tz_i = 3

    def __str__(self):
        return 'D2d'

    def S4(self, angle, data=None):
        angle = angle

        # first, bring an edge center to z-axis:

        def do_it(data):
            # do the actual rotation:
            data = self.S(angle=angle, data=data)
            return data

        if data is None:
            return do_it
        return do_it(data)

    def C2(self, angle, data=None):
        angle = angle

        def do_it(data):
            return self.rotate(angle=angle, data=data, axis='z')

        if data is None:
            return do_it
        return do_it(data)

    def C2p(self, angle=90., data=None):
        angle = angle

        def do_it(data):
            data = self.rotate(angle=angle, data=data, axis='z')
            data = self.rotate(angle=90., data=data, axis='x')

            # do the actual rotation:
            data = self.rotate(angle=180., data=data, axis='z')

            # rotate back:
            data = self.rotate(angle=-90., data=data, axis='x')
            data = self.rotate(angle=-angle, data=data, axis='z')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def sigma_d(self, angle, data=None):
        # first rotate so that the plane is xz plane, flip, and rotate back
        angle = angle

        def do_it(data):
            first = self.rotate(data=data, angle=angle)
            second = self.mirror_xz(first)
            third = self.rotate(data=second, angle=-angle)
            return third

        if data is None:
            return do_it
        return do_it(data)


class C2(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit),
                           ('C2', self.rotate_mainaxis(angle=180.))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = ['A', 'B']
        self.character_table = [[
            1.,
            1.,
        ], [1., -1.]]

        self.nof_operations = [1, 1]
        self.Tx_i = 1
        self.Ty_i = 1
        self.Tz_i = 0

    def __str__(self):
        return 'C2'


class Oh(Pointgroup):
    # main axis should be the z-axis!
    def __init__(self):
        self.operations = [('E', self.unit), ('1C3', self.C3(corner=0)),
                           ('2C3', self.C3(corner=1)),
                           ('3C3', self.C3(corner=2)),
                           ('4C3', self.C3(corner=3)),
                           ('5C3', self.C3(corner=4)),
                           ('6C3', self.C3(corner=5)),
                           ('7C3', self.C3(corner=6)),
                           ('8C3', self.C3(corner=7)),
                           ('1C2', self.C2(edge=0)), ('2C2', self.C2(edge=1)),
                           ('3C2', self.C2(edge=2)), ('4C2', self.C2(edge=3)),
                           ('5C2', self.C2(edge=4)), ('6C2', self.C2(edge=5)),
                           ('1C4', self.C4(face=0)), ('2C4', self.C4(face=1)),
                           ('3C4', self.C4(face=2)), ('4C4', self.C4(face=3)),
                           ('5C4', self.C4(face=4)), ('6C4', self.C4(face=5)),
                           ('1faceC2', self.faceC2(face=0)),
                           ('2faceC2', self.faceC2(face=1)),
                           ('3faceC2', self.faceC2(face=2)),
                           ('i', self.inversion), ('1S4', self.S4(face=0)),
                           ('2S4', self.S4(face=1)), ('3S4', self.S4(face=2)),
                           ('4S4', self.S4(face=3)), ('5S4', self.S4(face=4)),
                           ('6S4', self.S4(face=5)),
                           ('1S6', self.S6(corner=0)),
                           ('2S6', self.S6(corner=1)),
                           ('3S6', self.S6(corner=2)),
                           ('4S6', self.S6(corner=3)),
                           ('5S6', self.S6(corner=4)),
                           ('6S6', self.S6(corner=5)),
                           ('7S6', self.S6(corner=6)),
                           ('8S6', self.S6(corner=7)),
                           ('1sigma_h1', self.sigma_h(face=0)),
                           ('2sigma_h1', self.sigma_h(face=1)),
                           ('3sigma_h1', self.sigma_h(face=2)),
                           ('1sigma_d1', self.sigma_d(face=0)),
                           ('2sigma_d1', self.sigma_d(face=1)),
                           ('3sigma_d1', self.sigma_d(face=2)),
                           ('4sigma_d1', self.sigma_d(face=3)),
                           ('5sigma_d1', self.sigma_d(face=4)),
                           ('6sigma_d1', self.sigma_d(face=5))]
        self.operation_names = [pair[0] for pair in self.operations]
        self.symmetries = [
            'A1g', 'A2g', 'Eg', 'T1g', 'T2g', 'A1u', 'A2u', 'Eu', 'T1u', 'T2u'
        ]
        self.character_table = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                [1., 1., -1., -1., 1., 1., -1., 1., 1., -1.],
                                [2., -1., 0., 0., 2., 2., 0., -1., 2., 0.],
                                [3., 0., -1., 1., -1., 3., 1., 0., -1., -1.],
                                [3., 0., 1., -1., -1., 3., -1., 0., -1., 1.],
                                [1., 1., 1., 1., 1., -1., -1., -1., -1., -1.],
                                [1., 1., -1., -1., 1., -1., 1., -1., -1., 1.],
                                [2., -1., 0., 0., 2., -2., 0., 1., -2., 0.],
                                [3., 0., -1., 1., -1., -3., -1., 0., 1., 1.],
                                [3., 0., 1., -1., -1., -3., 1., 0., 1., -1.]]
        self.nof_operations = [1, 8, 6, 6, 3, 1, 6, 8, 3, 6]
        self.Tx_i = 8
        self.Ty_i = 8
        self.Tz_i = 8

    def __str__(self):
        return 'Oh'

    def C3(self, corner, data=None):
        corner = corner
        t1, t2 = divmod(corner, 4)
        t1 = (-1)**t1
        angle1 = (t2 * 90.) + 45.  # y
        angle2 = t1 * (np.arctan(1 / np.sqrt(2.)) * 180. / np.pi)  # x

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='y')
            data = self.rotate(angle=angle2, data=data, axis='x')

            data = self.rotate(angle=120., data=data, axis='z')

            data = self.rotate(angle=-angle2, data=data, axis='x')
            data = self.rotate(angle=-angle1, data=data, axis='y')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def C2(self, edge, data=None):
        edge = edge
        t1, t2 = divmod(edge, 3)
        angle1 = t1 * 90.
        set2 = [[45., 'x'], [45., 'y'], [-45., 'x']][t2]

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='y')
            data = self.rotate(angle=set2[0], data=data, axis=set2[1])

            data = self.rotate(angle=180., data=data, axis='z')

            data = self.rotate(angle=-set2[0], data=data, axis=set2[1])
            data = self.rotate(angle=-angle1, data=data, axis='y')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def C4(self, face, data=None):
        face = face
        t1, t2 = divmod(face, 3)
        axis = ['x', 'y', 'z'][t2]
        angle = (-1)**t1 * 90.

        def do_it(data):
            data = self.rotate(angle=90., data=data, axis=axis)

            data = self.rotate(angle=angle, data=data, axis='z')

            data = self.rotate(angle=-90., data=data, axis=axis)
            return data

        if data is None:
            return do_it
        return do_it(data)

    def faceC2(self, face, data=None):
        face = face
        t1, t2 = divmod(face, 3)
        axis = ['x', 'y', 'z'][t2]
        angle = 180.

        def do_it(data):
            data = self.rotate(angle=90., data=data, axis=axis)

            data = self.rotate(angle=angle, data=data, axis='y')

            data = self.rotate(angle=-90., data=data, axis=axis)
            return data

        if data is None:
            return do_it
        return do_it(data)

    def inversion(self, data=None):
        return self.invert(data)

    def S4(self, face, data=None):
        face = face
        t1, t2 = divmod(face, 3)
        axis = ['x', 'y', 'z'][t2]
        angle = (-1)**t1 * 90.

        def do_it(data):
            data = self.rotate(angle=90., data=data, axis=axis)

            data = self.S(angle=angle, data=data)

            data = self.rotate(angle=-90., data=data, axis=axis)
            return data

        if data is None:
            return do_it
        return do_it(data)

    def S6(self, corner, data=None):
        corner = corner
        t1, t2 = divmod(corner, 4)
        t1 = (-1)**t1
        angle1 = (t2 * 90.) + 45.  # y
        angle2 = t1 * (np.arctan(1 / np.sqrt(2.)) * 180. / np.pi)  # x

        def do_it(data):
            data = self.rotate(angle=angle1, data=data, axis='y')
            data = self.rotate(angle=angle2, data=data, axis='x')

            data = self.S(angle=60., data=data)

            data = self.rotate(angle=-angle2, data=data, axis='x')
            data = self.rotate(angle=-angle1, data=data, axis='y')
            return data

        if data is None:
            return do_it
        return do_it(data)

    def sigma_h(self, face, data=None):
        face = face
        t1, t2 = divmod(face, 3)
        axis = ['x', 'y', 'z'][t2]

        def do_it(data):
            data = self.rotate(angle=90., data=data, axis=axis)

            data = self.mirror_xy(data)

            data = self.rotate(angle=-90., data=data, axis=axis)
            return data

        if data is None:
            return do_it
        return do_it(data)

    def sigma_d(self, face, data=None):
        face = face
        t1, t2 = divmod(face, 3)
        axis = ['x', 'y', 'z'][t2]
        angle2 = (-1)**t1 * 45.

        def do_it(data):
            data = self.rotate(angle=90., data=data, axis=axis)
            data = self.rotate(angle=angle2, data=data, axis='z')

            data = self.mirror_xz(data)

            data = self.rotate(angle=-angle2, data=data, axis='z')
            data = self.rotate(angle=-90., data=data, axis=axis)
            return data

        if data is None:
            return do_it
        return do_it(data)
