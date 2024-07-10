import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import time
tic = time.time()
print('start')
comp = ['A', 'B', 'Va']
p_in = {'A,B/Va': [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'A/Va,B': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}
n_SL = 3
R = 8.314
m = [1, 2, 3]

G_end_mbrs_phase = np.zeros((3,3,3))

def xlogx(x):
    if x > 1e-12:
        return x * np.log(x)
    else:
        return -2.8e-11

def separate_nonzero_vector(vector):
    separated_vectors = []
    for i in range(len(vector)):
        if vector[i] == 1:
            new_vector = np.zeros_like(vector)
            new_vector[i] = 1
            separated_vectors.append(new_vector)
    return separated_vectors

def generalized_sum(G_end_mbrs_phase, ys):
    n = len(ys[0])  # Number of multipliers
    indices = range(len(ys))

    def recursive_sum(indices_list):
        if len(indices_list) == n:
            product_terms = [ys[idx][i] for idx, i in zip(indices_list, range(n))]
            G_term = G_end_mbrs_phase[tuple(indices_list)]
            return G_term * np.prod(product_terms)
        else:
            total = 0
            for idx in indices:
                total += recursive_sum(indices_list + [idx])
            return total

    return recursive_sum([])

def fpol(T, p, y1, y2):
    return (p[0] + p[1] * T) * (y1 * y2 * (y1 - y2) ** (p[2]))

def L(T, S, y1, y2, params_in):
    try:
        p = params_in[S]
    except KeyError:
        p = [[0, 0, 0]]
    return sum([fpol(T, pi, y1, y2) for pi in p])

def funct_to_optimize(T, y):
    ys = np.reshape(y, (3, n_SL))
  
    
    XS_lst = [L(T, '{},{}/{}'.format(comp[i], comp[j], comp[k]).encode('utf-8'), ys[i][si], ys[j][si], p_in)
                for i in range(len(comp)) for j in range(len(comp)) if i != j
                for k in range(len(comp)) for si in range(n_SL)
                ] + [L(T, '{}/{},{}'.format(comp[i], comp[j], comp[k]).encode('utf-8'), ys[j][si], ys[k][si], p_in)
                    for i in range(len(comp)) for j in range(len(comp))
                    for k in range(len(comp)) if j != k for si in range(n_SL)]
    G_XS = sum(XS_lst)

    G_conf = R*T*(m[0]*xlogx(ys[0][0]) +
                  m[0]*xlogx(ys[1][0]) +
                  m[0]*xlogx(ys[2][0]) + 
                  m[1]*xlogx(ys[0][1]) + 
                  m[1]*xlogx(ys[1][1]) + 
                  m[1]*xlogx(ys[2][1]) )

    G_ref = generalized_sum(G_end_mbrs_phase, ys)
    return G_ref + G_conf + G_XS


def run_minimization(T, ZA, ZB, ZVa):

    # Contraints for the site-fractions
    constraint_list = []
    lim_dnx = []
    lim_upx = []

    # Constraints: sum = 1
    for j in range(len(m)):
        lim_dnx.append(1 - 1e-7)
        lim_upx.append(1 + 1e-7)
        constraint_list.append(np.array([1 if (i + j) % len(m) == 0 else 0 for i in range(len(m) * len(comp))]))

    # Constraints: z=zum(my)/sum(m)
    C2, C3, C4 = np.zeros(len(m) * len(comp)), np.zeros(len(m) * len(comp)), np.zeros(len(m) * len(comp))
    fm = [mi / (sum(m)) for mi in m]
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
    vector_ones = [1, 1, 0, 1, 0, 1, 0, 1, 1]#[::-1]
    filtered_separated_vectors = separate_nonzero_vector(vector_ones)

    for vec in filtered_separated_vectors:
        
        constraint_list.append(vec)
    constraints_matrix = np.array(constraint_list)


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
            y0 = [0,0,0, 0,0,0, 0,0,0]

            constraint = LinearConstraint(constraints_matrix, lim_dn, lim_up)

            result = minimize(f2min, y0, bounds=Bounds([0] * len(y0), [1] * len(y0)), constraints=constraint,
                                method='COBYLA')
    return result

ys = np.array([.1,.2,.3,.11,.12,.13,.21,.22,.23])


XB_ = np.linspace(1e-7, 1-1e-7, 20)
ZB_ = np.linspace(0.2, 0.9, 20)
ZA_ = np.empty_like(ZB_)
ZVa_ = np.empty_like(ZB_)
for i in range(len(ZB_)):
    ZA_[i] = ZB_[i]*(1-XB_[i])/XB_[i]
    ZVa_[i] = 1 - ZA_[i] - ZB_[i]

T_ = np.linspace(300, 1000, 50)

result = run_minimization(T_, ZA_, ZB_, ZVa_)
print('end')
print(f"Elapsed time: {time.time() - tic:.2f} s")