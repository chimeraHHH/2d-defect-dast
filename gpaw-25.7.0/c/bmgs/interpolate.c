/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "../extensions.h"
#include "bmgs.h"

#ifdef K

void
IP1D(const T* a, const int n, const int m, T* restrict b, int skip[2])
{
    a += K / 2 - 1;

    for (int j = 0; j < m; j++) {
        const T* aa = a + j * (K - 1 - skip[1] + n);
        T* restrict bb = b + j;

        for (int i = 0; i < n; i++) {
            if (i == 0 && skip[0])
                bb -= m;
            else
                bb[0] = aa[0];

            if (i == n - 1 && skip[1])
                bb -= m;
            else {
                if (K == 2)
                    bb[m] = 0.5 * (aa[0] + aa[1]);
                else if (K == 4)
                    bb[m] = ( 0.5625 * (aa[ 0] + aa[1]) +
                             -0.0625 * (aa[-1] + aa[2]));
                else if (K == 6)
                    bb[m] = ( 0.58593750 * (aa[ 0] + aa[1]) +
                             -0.09765625 * (aa[-1] + aa[2]) +
                              0.01171875 * (aa[-2] + aa[3]));
                else
                    bb[m] = ( 0.59814453125 * (aa[ 0] + aa[1]) +
                             -0.11962890625 * (aa[-1] + aa[2]) +
                              0.02392578125 * (aa[-2] + aa[3]) +
                             -0.00244140625 * (aa[-3] + aa[4]));
            }
            aa++;
            bb += 2 * m;
        }
    }
}

#else
#  define K 2
#  define IP1D Z(bmgs_interpolate1D2)
#  include "interpolate.c"
#  undef IP1D
#  undef K
#  define K 4
#  define IP1D Z(bmgs_interpolate1D4)
#  include "interpolate.c"
#  undef IP1D
#  undef K
#  define K 6
#  define IP1D Z(bmgs_interpolate1D6)
#  include "interpolate.c"
#  undef IP1D
#  undef K
#  define K 8
#  define IP1D Z(bmgs_interpolate1D8)
#  include "interpolate.c"
#  undef IP1D
#  undef K

void
Z(bmgs_interpolate)(int k, int skip[3][2],
        const T* a, const int size[3], T* restrict b, T* restrict w)
{
    void (*ip)(const T*, int, int, T*, int[2]);
    int e;

    if (k == 2)
        ip = Z(bmgs_interpolate1D2);
    else if (k == 4)
        ip = Z(bmgs_interpolate1D4);
    else if (k == 6)
        ip = Z(bmgs_interpolate1D6);
    else
        ip = Z(bmgs_interpolate1D8);

    e = k - 1;

    ip(a, size[2] - e + skip[2][1],
            size[0] *
            size[1],
            b, skip[2]);
    ip(b, size[1] - e + skip[1][1],
            size[0] *
            ((size[2] - e) * 2 - skip[2][0] + skip[2][1]),
            w, skip[1]);
    ip(w, size[0] - e + skip[0][1],
            ((size[1] - e) * 2 - skip[1][0] + skip[1][1]) *
            ((size[2] - e) * 2 - skip[2][0] + skip[2][1]),
            b, skip[0]);
}

#endif
