from dnnv.properties import *
import numpy as np

N = Network("N")
ub = np.ones((1, 2))

Forall(x, Implies(abs(x) <= ub, N(x) >= 1))
