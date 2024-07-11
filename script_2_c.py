import numpy as np
from functions_cython import run_minimization
import time

tic = time.time()
print('start')

ys = np.array([.1, .2, .3, .11, .12, .13, .21, .22, .23])

XB_ = np.linspace(1e-7, 1-1e-7, 20)
ZB_ = np.linspace(0.2, 0.9, 20)
ZA_ = np.empty_like(ZB_)
ZVa_ = np.empty_like(ZB_)
for i in range(len(ZB_)):
    ZA_[i] = ZB_[i] * (1 - XB_[i]) / XB_[i]
    ZVa_[i] = 1 - ZA_[i] - ZB_[i]

T_ = np.linspace(300, 1000, 50)

result = run_minimization(T_, ZA_, ZB_, ZVa_)
print('end')
print(f"Elapsed time: {time.time() - tic:.2f} s")
