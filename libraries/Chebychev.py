import numpy as np


def Chebyshev_points(xmin, xmax, n):
    ns = np.arange(1, n + 1)
    x = np.cos((2 * ns - 1) * np.pi / (2 * n))
    return (xmin + xmax) / 2 + (xmax - xmin) * x / 2


def eval_Barycentric(v, c, x):
    """"""
    e1 = 0
    e2 = 0
    for j in range(len(c)):
        if x == c[j]:
            return v[j]
        else:
            if j == 0 or j == len(c) - 1:
                e1 += 0.5 * ((-1) ** j * v[j]) / (x - c[j])  # A : changed by Amine
                e2 += 0.5 * ((-1) ** j) / (x - c[j])
            else:
                e1 += ((-1) ** j * v[j]) / (x - c[j])
                e2 += ((-1) ** j) / (x - c[j])
                # e1+= 0.5*((-1)**j * v[j])/(x-c[j])      # A : what was here. Doesn't take into account that we multiply by 0.5
            # e2+= 0.5*((-1)**j)/(x-c[j])                # only on the first and last terms of the sum
    Pn = e1 / e2
    return Pn

