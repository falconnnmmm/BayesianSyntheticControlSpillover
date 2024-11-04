# NOTE: This is a simple example of how to use the BayesianSyntheticControl class.
# NOTE: This is not a part of the package and is ONLY for demonstration purposes.
# NOTE: Thus, the DGP in this file is not the same as the one used in the paper.

import numpy as np
import pandas as pd

from modules._bscm_sampling import BayesianSyntheticControl

N = 10
T = 20
k = 2
T0 = 10

control_outcome = np.random.randn(N, T)

weight = np.random.randn(N)

treatment_outcome = weight @ control_outcome

control_outcome = pd.DataFrame(control_outcome)
treatment_outcome = pd.DataFrame(treatment_outcome)
control_outcome["unit"] = np.arange(N)

control_outcome = control_outcome.melt(id_vars="unit", var_name="time")

treatment_outcome["time"] = np.arange(T)

X = pd.DataFrame(np.random.randn(N*T, k))
X.columns = ["X1", "X2"]
X["unit"] = np.tile(np.arange(N), T)
X["time"] = np.repeat(np.arange(T), N)

adj_mat = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        adj_mat[i, j] = np.random.randint(0, 2)
adj_vec = np.random.randint(0, 2, N)

bsc = BayesianSyntheticControl(control_outcome, treatment_outcome, X, adj_mat, adj_vec, T0, "time", "unit")

bsc.mcmc()
te, spillover = bsc.calculate_effects()



