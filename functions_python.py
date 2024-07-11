import numpy as np
def xlogx(x: float) -> float:
    if x > 1e-12:
        return x * np.log(x)
    else:
        return -2.8e-11