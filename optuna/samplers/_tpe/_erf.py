# This code is the modified version of erf function in FreeBSD's standard C library.
# origin: FreeBSD /usr/src/lib/msun/src/s_erf.c
# https://github.com/freebsd/freebsd-src/blob/main/lib/msun/src/s_erf.c

# /* @(#)s_erf.c 5.1 93/09/24 */
# /*
#  * ====================================================
#  * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#  *
#  * Developed at SunPro, a Sun Microsystems, Inc. business.
#  * Permission to use, copy, modify, and distribute this
#  * software is freely granted, provided that this notice
#  * is preserved.
#  * ====================================================
#  */

import math

import numpy as np
from numpy.polynomial import Polynomial


erx = 8.45062911510467529297e-01
# /*
#  * In the domain [0, 2**-28], only the first term in the power series
#  * expansion of erf(x) is used.  The magnitude of the first neglected
#  * terms is less than 2**-84.
#  */
efx = 1.28379167095512586316e-01

# Coefficients for approximation to erf on [0,0.84375]

pp0 = 1.28379167095512558561e-01
pp1 = -3.25042107247001499370e-01
pp2 = -2.84817495755985104766e-02
pp3 = -5.77027029648944159157e-03
pp4 = -2.37630166566501626084e-05
pp = Polynomial([pp0, pp1, pp2, pp3, pp4])
qq1 = 3.97917223959155352819e-01
qq2 = 6.50222499887672944485e-02
qq3 = 5.08130628187576562776e-03
qq4 = 1.32494738004321644526e-04
qq5 = -3.96022827877536812320e-06
qq = Polynomial([1.0, qq1, qq2, qq3, qq4, qq5])

# Coefficients for approximation to erf in [0.84375,1.25]

pa0 = -2.36211856075265944077e-03
pa1 = 4.14856118683748331666e-01
pa2 = -3.72207876035701323847e-01
pa3 = 3.18346619901161753674e-01
pa4 = -1.10894694282396677476e-01
pa5 = 3.54783043256182359371e-02
pa6 = -2.16637559486879084300e-03
pa = Polynomial([pa0, pa1, pa2, pa3, pa4, pa5, pa6])
qa1 = 1.06420880400844228286e-01
qa2 = 5.40397917702171048937e-01
qa3 = 7.18286544141962662868e-02
qa4 = 1.26171219808761642112e-01
qa5 = 1.36370839120290507362e-02
qa6 = 1.19844998467991074170e-02
qa = Polynomial([1.0, qa1, qa2, qa3, qa4, qa5, qa6])

# Coefficients for approximation to erfc in [1.25,1/0.35]

ra0 = -9.86494403484714822705e-03
ra1 = -6.93858572707181764372e-01
ra2 = -1.05586262253232909814e01
ra3 = -6.23753324503260060396e01
ra4 = -1.62396669462573470355e02
ra5 = -1.84605092906711035994e02
ra6 = -8.12874355063065934246e01
ra7 = -9.81432934416914548592e00
ra = Polynomial([ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7])
sa1 = 1.96512716674392571292e01
sa2 = 1.37657754143519042600e02
sa3 = 4.34565877475229228821e02
sa4 = 6.45387271733267880336e02
sa5 = 4.29008140027567833386e02
sa6 = 1.08635005541779435134e02
sa7 = 6.57024977031928170135e00
sa8 = -6.04244152148580987438e-02
sa = Polynomial([1.0, sa1, sa2, sa3, sa4, sa5, sa6, sa7, sa8])

# Coefficients for approximation to erfc in [1/.35,28]

rb0 = -9.86494292470009928597e-03
rb1 = -7.99283237680523006574e-01
rb2 = -1.77579549177547519889e01
rb3 = -1.60636384855821916062e02
rb4 = -6.37566443368389627722e02
rb5 = -1.02509513161107724954e03
rb6 = -4.83519191608651397019e02
rb = Polynomial([rb0, rb1, rb2, rb3, rb4, rb5, rb6])
sb1 = 3.03380607434824582924e01
sb2 = 3.25792512996573918826e02
sb3 = 1.53672958608443695994e03
sb4 = 3.19985821950859553908e03
sb5 = 2.55305040643316442583e03
sb6 = 4.74528541206955367215e02
sb7 = -2.24409524465858183362e01
sb = Polynomial([1.0, sb1, sb2, sb3, sb4, sb5, sb6, sb7])


def _erf_right(x_sorted: np.ndarray) -> np.ndarray:
    # x_sorted must be a sorted 1D non-negative value array.
    assert len(x_sorted.shape) == 1, "x_sorted must be flattened."
    boundary_indices = np.searchsorted(x_sorted, [2**-28, 0.84375, 1.25, 6])

    def calc_case_small1(x: np.ndarray) -> np.ndarray:
        z = x * x
        return x * (1 + pp(z) / qq(z))

    def calc_case_small2(x: np.ndarray) -> np.ndarray:
        s = x - 1.0
        return erx + pa(s) / qa(s)

    def calc_case_med(x: np.ndarray) -> np.ndarray:
        x2 = x * x
        s = 1.0 / x2
        mid_idx = np.searchsorted(x, 1 / 0.35)
        Q = np.empty_like(s)
        Q[:mid_idx] = ra(s[:mid_idx]) / sa(s[:mid_idx])
        Q[mid_idx:] = rb(s[mid_idx:]) / sb(s[mid_idx:])
        # the following 3 lines are omitted for the following reasons:
        # (1) there are no easy way to implement SET_LOW_WORD equivalent method in NumPy
        # (2) we don't need very high accuracy in our use case.
        # z = x
        # SET_LOW_WORD(z, 0)
        # r = np.exp(-z * z - 0.5625) * np.exp((z - x) * (z + x) + Q)
        return 1.0 - np.exp(-x2 - 0.5625 + Q) / x

    out = np.where(np.isnan(x_sorted), np.nan, 1.0)  # Big values will receive 1.0.
    if (tiny_end := boundary_indices[0]) > 0:
        out[:tiny_end] = x_sorted[:tiny_end] * (1 + efx)
    if (small1_end := boundary_indices[1]) - tiny_end > 0:
        out[tiny_end:small1_end] = calc_case_small1(x_sorted[tiny_end:small1_end])
    if (small2_end := boundary_indices[2]) - small1_end > 0:
        out[small1_end:small2_end] = calc_case_small2(x_sorted[small1_end:small2_end])
    if (med_end := boundary_indices[3]).size:
        out[small2_end:med_end] = calc_case_med(x_sorted[small2_end:med_end])

    return out


def erf(x: np.ndarray) -> np.ndarray:
    if x.size < 2000:
        return np.asarray([math.erf(v) for v in x.ravel()]).reshape(x.shape)

    a = np.abs(x.ravel())
    order = np.argsort(a)
    rev = np.empty(a.size, dtype=int)
    rev[order] = np.arange(a.size)
    return np.sign(x) * _erf_right(a[order])[rev].reshape(x.shape)
