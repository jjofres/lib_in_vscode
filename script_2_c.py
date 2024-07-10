import numpy as np
from optimization import run_minimization

# Example data for T, ZA, ZB, ZVa, XB
XB_ = np.linspace(1e-7, 1-1e-7, 20)
ZB_ = np.linspace(0.2, 0.9, 20)
ZA_ = np.empty_like(ZB_)
ZVa_ = np.empty_like(ZB_)
for i in range(len(ZB_)):
    ZA_[i] = ZB_[i]*(1-XB_[i])/XB_[i]
    ZVa_[i] = 1 - ZA_[i] - ZB_[i]

T_ = np.linspace(300, 1000, 50)

run_minimization(T_, ZA_, ZB_, ZVa_)