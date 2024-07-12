from libc.math cimport log

cpdef double xlogx(double x):
    if x > 1e-12:
        return x * log(x)
    else:
        return -2.8e-11

cpdef double fpol(double T, double[:] p, double y1, double y2):
    return (p[0] + p[1] * T) * (y1 * y2 * (y1 - y2) ** p[2])

from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np

cpdef list separate_nonzero_vector(cnp.ndarray[cnp.int_t, ndim=1] vecti):
    cdef cnp.ndarray[cnp.int_t, ndim=1] new_vecti
    cdef list separated_vectis = []
    cdef Py_ssize_t i, n = vecti.shape[0]

    for i in range(n):
        if vecti[i] == 1:
            new_vecti = np.zeros_like(vecti)
            new_vecti[i] = 1
            separated_vectis.append(new_vecti)

    return separated_vectis

cpdef double cL(double T, double y1, double y2, cnp.ndarray[cnp.double_t, ndim=2] p):
    cdef int i, n = p.shape[0]
    cdef double result = 0.0
    for i in range(n):
        result += fpol(T, p[i], y1, y2)
    return result

def calc_Gconf(double T, double[:] ys, double[:] m, int len_comp):
    cdef int si, i, isi
    cdef double ce = 0
    cdef double G_conf
    cdef double R = 8.314
    cdef int len_m = len(m)

    for si in range(len_m):
        for i in range(len_comp):
            isi = i * len_m + si
            ce += m[si] * xlogx(ys[isi])

    G_conf = R * T * ce / sum(m)
    return G_conf

