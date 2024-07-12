import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from functions_cython import xlogx, separate_nonzero_vector, cL, calc_Gconf, fpol
from numpy import array as nparray

comp = ['A', 'B', 'Va']
len_comp = len(comp)
p_in = {'A,B/Va': [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], 'A/Va,B': [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        'A,B/A':[[0.,0.,0.]], 'A/B,A':[[0.,0.,0.]], 'A,B/B':[[0.,0.,0.]],'A/B,B':[[0.,0.,0.]],
        'A/B,Va':[[0.,0.,0.]],'A,Va/A':[[0.,0.,0.]],'A/Va,A':[[0.,0.,0.]],'A,Va/B':[[0.,0.,0.]],
        'A,Va/Va':[[0.,0.,0.]],'A/Va,Va':[[0.,0.,0.]],'B,A/A':[[0.,0.,0.]],'B/A,A':[[0.,0.,0.]],
        'B,A/B':[[0.,0.,0.,]],'B/A,B':[[0.,0.,0.,]],'B,A/Va':[[0.,0.,0.]],'B/A,Va':[[0.,0.,0.]],
        'B,Va/A':[[0.,0.,0.]], 'B/Va,A':[[0.,0.,0.]],'B,Va/B':[[0.,0.,0.]],'B/Va,B':[[0.,0.,0.]],
        'B,Va/Va':[[0.,0.,0.]],'B/Va,Va':[[0.,0.,0.]],'Va,A/A':[[0.,0.,0.]],'Va/A,A':[[0.,0.,0.]],
        'Va,A/B':[[0.,0.,0.]],'Va/A,B':[[0.,0.,0.]],'Va,A/Va':[[0.,0.,0.]],'Va/A,Va':[[0.,0.,0.]],
        'Va,B/A':[[0.,0.,0.]],'Va/B,A':[[0.,0.,0.]],'Va,B/B':[[0.,0.,0.]],'Va/B,B':[[0.,0.,0.]],
        'Va,B/Va':[[0.,0.,0.]],'Va/B,Va':[[0.,0.,0.]]}
n_SL = 3
R = 8.314
m = [1., 2., 3.]
len_m = len(m)

G_end_mbrs_phase = np.zeros((3, 3, 3))


def L(T: float, S: str, y1: float, y2: float, params_in: dict[str, list]) -> float:
    p = nparray(params_in[S])
    return cL(T, y1, y2, p)


def calc_GXS(T, ys):
    G_XS = 0
    range_len_comp = range(len_comp)
    range_n_SL = range(n_SL)
    for i in range_len_comp:
        for j in range_len_comp:
            if i != j:
                for k in range_len_comp:
                    for si in range_n_SL:
                        isi = i * len_m + si
                        jsi = j * len_m + si
                        ksi = k * len_m + si
                        G_XS += L(T, comp[i] + ',' + comp[j] + '/' + comp[k], ys[isi], ys[jsi], p_in)
                        G_XS += L(T, comp[i] + '/' + comp[j] + ',' + comp[k], ys[jsi], ys[ksi], p_in)
    return G_XS

gen_ix = lambda i, j, c: i*c+j
from functools import reduce
from operator import mul
def calc_Gref(ys, G_end_mbrs_phase, m):
    G_ref = 0

    for ix, val in np.ndenumerate(G_end_mbrs_phase):
        def gen_ix_simpl(i):
            return ys[gen_ix(ix[i], i, len_m)]
        ys2prod = map(gen_ix_simpl, range(len(ix)))
        p = reduce(mul, ys2prod, 1)
        G_ref += val * p
    return G_ref


def funct_to_optimize(T: float, ys: list) -> float:
    if False:
        G_XS = calc_GXS(T, ys)
    else:
        G_XS = 0
    G_conf = calc_Gconf(T, nparray(ys), nparray(m), len_comp)

    G_ref = calc_Gref(ys, G_end_mbrs_phase, m)

    return G_ref + G_conf + G_XS


def min_contraints() -> tuple[list[list[float]], list[float], list[float], list[list[int]]]:
    # Contraints for the site-fractions
    constraint_list = []
    lim_dnx = []
    lim_upx = []

    # Constraints: sum = 1
    for j in range(len_m):
        lim_dnx.append(1 - 1e-7)
        lim_upx.append(1 + 1e-7)
        arr_lst = np.zeros(len_m * len_comp)
        for i in range(len_m * len_comp):
            if (i + j) % len_m == 0:
                arr_lst[i] = 1
        constraint_list.append(arr_lst)

    # Constraints: z=zum(my)/sum(m)
    C2 = np.zeros(len_m * len_comp)
    C3 = np.zeros(len_m * len_comp)
    C4 = np.zeros(len_m * len_comp)
    fm = []
    summ = sum(m)
    for mi in m:
        fm.append(mi / summ)

    for i in range(len_comp):
        for j in range(len_m):
            ix = i * len_m + (j)
            if i == 0:
                C2[ix] = fm[j]
            elif i == 1:
                C3[ix] = fm[j]
            elif i == 2:
                C4[ix] = fm[j]
    constraint_list.append(C2)
    constraint_list.append(C3)
    constraint_list.append(C4)

    # Constrints: non-present species means their respective sitefraction is zero.
    vector_ones = nparray([1, 1, 0, 1, 0, 1, 0, 1, 1])  # [::-1]
    filtered_separated_vectors = separate_nonzero_vector(vector_ones)

    for vec in filtered_separated_vectors:
        constraint_list.append(vec)
    # constraints_matrix = nparray(constraint_list)

    return constraint_list, lim_dnx, lim_upx, filtered_separated_vectors


def run_minimization(T: float, ZA: np.ndarray, ZB: np.ndarray, ZVa: np.ndarray) -> None:
    constraint_list, lim_dnx, lim_upx, filtered_separated_vectors = min_contraints()

    # hola
    rz = range(len(ZA))
    for Ti in T:
        # print('Temperature:', Ti)
        for i in rz:
            ZAi, ZBi, ZVai = ZA[i], ZB[i], ZVa[i]
            # print('Composition:', XBi)
            lim_up = lim_upx.copy()
            lim_dn = lim_dnx.copy()
            lim_dn_append = lim_dn.append
            lim_up_append = lim_up.append
            for zz in [ZAi, ZBi, ZVai]:
                lim_dn_append(zz - 1e-7)
                lim_up_append(zz + 1e-7)
            for _ in filtered_separated_vectors:
                lim_up_append(1e-7)
                lim_dn_append(-1e-7)

            f2min = lambda ys, Tx=Ti: funct_to_optimize(Tx, ys)
            y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            constraint = LinearConstraint(constraint_list, lim_dn, lim_up)

            result = minimize(f2min, y0, bounds=Bounds([0] * len(y0), [1] * len(y0)), constraints=constraint,
                              method='COBYLA')

    # return result