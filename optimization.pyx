import numpy as np
cimport numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import time

cdef double xlogx(double x):
    if x > 1e-12:
        return x * np.log(x)
    else:
        return -2.8e-11

cdef list separate_nonzero_vector(np.ndarray[double, ndim=1] vector):
    cdef list separated_vectors = []
    cdef int i
    cdef np.ndarray[double, ndim=1] new_vector
    for i in range(len(vector)):
        if vector[i] == 1:
            new_vector = np.zeros_like(vector)
            new_vector[i] = 1
            separated_vectors.append(new_vector)
    return separated_vectors

def generalized_sum(G_end_mbrs_phase, ys):
    cdef int n = len(ys[0])
    cdef int idx
    cdef double total = 0

    def recursive_sum(indices_list):
        cdef int i
        cdef double product_terms
        cdef np.ndarray[double, ndim=1] indices = np.array(indices_list, dtype=np.int32)
        if len(indices_list) == n:
            product_terms = [ys[idx][i] for idx, i in zip(indices_list, range(n))]
            G_term = G_end_mbrs_phase[tuple(indices)]
            return G_term * np.prod(product_terms)
        else:
            total = 0
            for idx in range(len(ys)):
                total += recursive_sum(indices_list + [idx])
            return total

    return recursive_sum([])

cdef double fpol(double T, np.ndarray[double, ndim=1] p, double y1, double y2):
    return (p[0] + p[1] * T) * (y1 * y2 * (y1 - y2) ** (p[2]))

cdef double L(double T, char* S, double y1, double y2, dict params_in):
    try:
        p = params_in[S]
    except KeyError:
        p = [[0, 0, 0]]
    return sum([fpol(T, np.array(pi, dtype=np.float64), y1, y2) for pi in p])

def funct_to_optimize(double T, np.ndarray[double, ndim=1] y, list comp, dict p_in):
    cdef int i, j, k, si
    cdef np.ndarray[double, ndim=2] ys = np.reshape(y, (3, 3))
    cdef list XS_lst = [L(T, '{}, {}/{}'.format(comp[i], comp[j], comp[k]).encode('utf-8'), ys[i][si], ys[j][si], p_in)
                for i in range(len(comp)) for j in range(len(comp)) if i != j
                for k in range(len(comp)) for si in range(3)
                ] + [L(T, '{}/ {},{}'.format(comp[i], comp[j], comp[k]).encode('utf-8'), ys[j][si], ys[k][si], p_in)
                    for i in range(len(comp)) for j in range(len(comp))
                    for k in range(len(comp)) for si in range(3)]
    return sum(XS_lst)

def run_minimization(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=1] ZA, np.ndarray[double, ndim=1] ZB, np.ndarray[double, ndim=1] ZVa):
    cdef list comp = ['A', 'B', 'Va']
    cdef dict p_in = {'A,B/Va': [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'A/Va,B': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}
    cdef int n_SL = 3
    cdef double R = 8.314
    cdef list m = [1, 2, 3]
    cdef np.ndarray[double, ndim=3] G_end_mbrs_phase = np.zeros((3, 3, 3), dtype=np.float64)
    cdef list constraint_list = []
    cdef np.ndarray[double, ndim=1] lim_upx = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    cdef np.ndarray[double, ndim=1] lim_dnx = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cdef np.ndarray[double, ndim=2] constraints_matrix

    # Initialize constraints_matrix and constraint_list
    vector_ones = [1, 1, 0, 1, 0, 1, 0, 0, 0]
    filtered_separated_vectors = separate_nonzero_vector(np.array(vector_ones, dtype=np.float64))
    for vec in filtered_separated_vectors:
        constraint_list.append(vec)
    constraints_matrix = np.array(constraint_list, dtype=np.float64)

    for Ti in T:
        for ZAi, ZBi, ZVai in zip(ZA, ZB, ZVa):
            lim_up = lim_upx.copy()
            lim_dn = lim_dnx.copy()
            for zz in [ZAi, ZBi, ZVai]:
                lim_dn = np.append(lim_dn, zz - 1e-7)
                lim_up = np.append(lim_up, zz + 1e-7)
            for _ in range(len(comp)):
                lim_up = np.append(lim_up, 1e-7)
                lim_dn = np.append(lim_dn, -1e-7)

            f2min = lambda ys, Tx=Ti: funct_to_optimize(Tx, ys, comp, p_in)
            y0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
            constraint = LinearConstraint(constraints_matrix, lim_dn, lim_up)
            result = minimize(f2min, y0, bounds=Bounds([0] * len(y0), [1] * len(y0)), constraints=constraint, method='COBYLA')
    return result
