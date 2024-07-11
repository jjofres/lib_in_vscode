import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from functions_cython import xlogx, separate_nonzero_vector, cL

comp = ['A', 'B', 'Va']
p_in = {'A,B/Va': [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], 'A/Va,B': [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]}
n_SL = 3
R = 8.314
m = [1, 2, 3]

G_end_mbrs_phase = np.zeros((3, 3, 3))


def L(T: float, S: str, y1: float, y2: float, params_in: dict[str, list]) -> float:
    try:
        p = np.array(params_in[S])
    except KeyError:
        p = np.array([[0., 0., 0.]])

    return cL(T, y1, y2, p)#sum([fpol(T, np.array(pi), y1, y2) for pi in p])


def funct_to_optimize(T: float, ys: list) -> float:
    G_XS = 0
    for i in range(len(comp)):
        for j in range(len(comp)):
            if i != j:
                for k in range(len(comp)):
                    for si in range(n_SL):
                        isi = i * len(m) + si
                        jsi = j * len(m) + si
                        ksi = k * len(m) + si
                        G_XS += L(T, comp[i]+','+comp[j]+'/'+comp[k], ys[isi], ys[jsi], p_in)
                        G_XS += L(T, comp[i]+'/'+comp[j]+','+comp[k], ys[jsi], ys[ksi], p_in)

    ce = 0
    for si in range(len(m)):
        for i in range(len(comp)):
            isi = i * len(m) + si
            ce += m[si] * xlogx(ys[isi])
    G_conf = R * T * ce / sum(m)

    G_ref = 0
    for ix, val in np.ndenumerate(G_end_mbrs_phase):
        ys2prod = []
        for i in range(len(ix)):
            ii = ix[i] * len(m) + i
            ys2prod.append(ys[ii])
        # ys2prod = [y[ix[i]][i] for i in range(len(ix))]
        # p = np.prod(ys2prod)
        p = 1
        for yp in ys2prod:
            p = p * yp
        G_ref += val * p

    return G_ref + G_conf + G_XS


def min_contraints() -> tuple[list[list[float]], list[float], list[float], list[list[int]]]:
    # Contraints for the site-fractions
    constraint_list = []
    lim_dnx = []
    lim_upx = []

    # Constraints: sum = 1
    for j in range(len(m)):
        lim_dnx.append(1 - 1e-7)
        lim_upx.append(1 + 1e-7)
        arr_lst = np.zeros(len(m) * len(comp))
        for i in range(len(m) * len(comp)):
            if (i + j) % len(m) == 0:
                arr_lst[i] = 1
        constraint_list.append(arr_lst)

    # Constraints: z=zum(my)/sum(m)
    C2 = np.zeros(len(m) * len(comp))
    C3 = np.zeros(len(m) * len(comp))
    C4 = np.zeros(len(m) * len(comp))
    fm = []
    summ = sum(m)
    for mi in m:
        fm.append(mi / summ)

    for i in range(len(comp)):
        for j in range(len(m)):
            ix = i * len(m) + (j)
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
    vector_ones = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1])  # [::-1]
    filtered_separated_vectors = separate_nonzero_vector(vector_ones)

    for vec in filtered_separated_vectors:
        constraint_list.append(vec)
    # constraints_matrix = np.array(constraint_list)

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
            for zz in [ZAi, ZBi, ZVai]:
                lim_dn.append(zz - 1e-7)
                lim_up.append(zz + 1e-7)
            for _ in filtered_separated_vectors:
                lim_up.append(1e-7)
                lim_dn.append(-1e-7)

            f2min = lambda ys, Tx=Ti: funct_to_optimize(Tx, ys)
            y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            constraint = LinearConstraint(constraint_list, lim_dn, lim_up)

            result = minimize(f2min, y0, bounds=Bounds([0] * len(y0), [1] * len(y0)), constraints=constraint,
                              method='COBYLA')

    # return result