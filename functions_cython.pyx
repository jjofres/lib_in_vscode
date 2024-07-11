import numpy as np
cimport numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
cimport cython

cdef double xlogx(double x):
    if x > 1e-12:
        return x * np.log(x)
    else:
        return -2.8e-11

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list separate_nonzero_vector(np.ndarray[double, ndim=1] vector):
    cdef list separated_vectors = []
    cdef int i
    for i in range(len(vector)):
        if vector[i] == 1:
            new_vector = np.zeros_like(vector)
            new_vector[i] = 1
            separated_vectors.append(new_vector)
    return separated_vectors

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double fpol(double T, np.ndarray[double, ndim=1] p, double y1, double y2):
    return (p[0] + p[1] * T) * (y1 * y2 * (y1 - y2) ** p[2])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double L(double T, bytes S, double y1, double y2, dict params_in):
    cdef np.ndarray[double, ndim=1] p
    try:
        p = params_in[S]
    except KeyError:
        p = np.array([0, 0, 0], dtype=np.float64)
    return sum([fpol(T, pi, y1, y2) for pi in p])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double funct_to_optimize(double T, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=3] G_end_mbrs_phase,
                               list comp, int n_SL, double R, np.ndarray[double, ndim=1] m, dict p_in):
    cdef np.ndarray[double, ndim=2] ys = np.reshape(y, (3, n_SL))
    cdef list XS_lst = [L(T, '{},{}/{}'.format(comp[i], comp[j], comp[k]).encode('utf-8'), ys[i][si], ys[j][si], p_in)
                        for i in range(len(comp)) for j in range(len(comp)) if i != j
                        for k in range(len(comp)) for si in range(n_SL)
                        ] + [L(T, '{}/{},{}'.format(comp[i], comp[j], comp[k]).encode('utf-8'), ys[j][si], ys[k][si],
                               p_in)
                             for i in range(len(comp)) for j in range(len(comp))
                             for k in range(len(comp)) if j != k for si in range(n_SL)]
    G_XS = sum(XS_lst)

    G_conf = R * T * (m[0] * xlogx(ys[0][0]) +
                      m[0] * xlogx(ys[1][0]) +
                      m[0] * xlogx(ys[2][0]) +
                      m[1] * xlogx(ys[0][1]) +
                      m[1] * xlogx(ys[1][1]) +
                      m[1] * xlogx(ys[2][1]))

    G_ref = 0
    for ix, val in np.ndenumerate(G_end_mbrs_phase):
        ys2prod = [ys[ix[i]][i] for i in range(len(ix))]
        p = 1
        for yp in ys2prod:
            p *= yp
        G_ref += val * p

    return G_ref + G_conf + G_XS

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict run_minimization(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=1] ZA, np.ndarray[double, ndim=1] ZB,
                            np.ndarray[double, ndim=1] ZVa):
    cdef np.ndarray[double, ndim=3] G_end_mbrs_phase = np.zeros((3, 3, 3))
    cdef list comp = ['A', 'B', 'Va']
    cdef int n_SL = 3
    cdef double R = 8.314
    cdef np.ndarray[double, ndim=1] m = np.array([1, 2, 3], dtype=np.float64)
    cdef dict p_in = {'A,B/Va': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
                      'A/Va,B': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)}

    cdef list constraint_list = []
    cdef int i, j
    cdef np.ndarray[double, ndim=2] C2, C3, C4
    cdef np.ndarray[double, ndim=1] fm = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0], dtype=np.float64)
    cdef np.ndarray[double, ndim=2] constraints_matrix
    cdef np.ndarray[double, ndim=1] lim_upx = np.array([0, 0, 0], dtype=np.float64)
    cdef np.ndarray[double, ndim=1] lim_dnx = np.array([0, 0, 0], dtype=np.float64)
    cdef np.ndarray[double, ndim=1] lim_up, lim_dn

    for ix in range(3):
        C2 = np.zeros((3, 3), dtype=np.float64)
        C3 = np.zeros((3, 3), dtype=np.float64)
        C4 = np.zeros((3, 3), dtype=np.float64)
        for j in range(len(fm)):
            if ix == 0:
                C2[ix, j] = fm[j]
            elif ix == 1:
                C3[ix, j] = fm[j]
            elif ix == 2:
                C4[ix, j] = fm[j]
        constraint_list.append(C2)
        constraint_list.append(C3)
        constraint_list.append(C4)

    vector_ones = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.float64)
    filtered_separated_vectors = separate_nonzero_vector(vector_ones)

    for vec in filtered_separated_vectors:
        constraint_list.append(vec)
    constraints_matrix = np.vstack(constraint_list)  # Using vstack to ensure homogeneity

    results = {}
    for Ti in T:
        for i in range(len(ZA)):
            ZAi, ZBi, ZVai = ZA[i], ZB[i], ZVa[i]
            lim_up = lim_upx.copy()
            lim_dn = lim_dnx.copy()
            for zz in [ZAi, ZBi, ZVai]:
                lim_dn = np.append(lim_dn, zz - 1e-7)
                lim_up = np.append(lim_up, zz + 1e-7)
            for _ in filtered_separated_vectors:
                lim_up = np.append(lim_up, 1e-7)
                lim_dn = np.append(lim_dn, -1e-7)

            result = minimize(
                funct_to_optimize,
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
                args=(Ti, G_end_mbrs_phase, comp, n_SL, R, m, p_in),
                bounds=Bounds([0] * 9, [1] * 9),
                constraints=LinearConstraint(constraints_matrix, lim_dn, lim_up),
                method='COBYLA'
            )
            results[(Ti, ZAi, ZBi, ZVai)] = result

    return results
