import numpy as np
import pytorch_lightning as pl
from scipy import optimize
import sys
from .LQ import *

# Calculates the LQ solution imposing symmetry
# by hand in the optimization process
def investment_equilibrium_LQ(N, params):
    H_iv = [80.0, -0.2, 0.0]

    # Equation (22)
    B = np.zeros([N + 2, 1])
    B[1] = 1.0

    # Equation (23)
    C_1 = np.zeros([N + 1, N + 1])
    C_2 = np.zeros([1, N + 1])
    C_1[np.diag_indices(N + 1)] = params.sigma
    C_1[:, 0] = params.eta
    C_1[0, 1] = params.sigma
    C = np.concatenate((C_2, C_1))

    # Equation (24)
    R = np.zeros([N + 2, N + 2])
    R[1, :] = params.alpha_1 / (2 * N)
    R[:, 1] = params.alpha_1 / (2 * N)
    R[1, 1] = 0.0
    R[0, 1] = -params.alpha_0 / 2
    R[1, 0] = -params.alpha_0 / 2

    Q = params.gamma / 2

    # calculating A_hat
    def F_root(H):
        # Equation (30)
        H_0, H_1, H_2 = H  # H_2 not used

        # Equation (21)
        A = (H_1 / N) * np.ones([N + 2, N + 2])
        A[np.diag_indices(N + 2)] = 1.0 - params.delta + H_1 / N
        A[:, 0] = H_0
        A[:, 1] = 0.0
        A[0, :] = 0.0
        A[1, :] = 0.0
        A[0, 0] = 1.0
        A[1, 1] = 1.0 - params.delta

        lq = LQ(Q, R, A, B, C, beta=params.beta)
        P, F, d = lq.stationary_values()
        return np.array([F[0][0], F[0][1], F[0][2]]) - np.array([-H[0], 0.0, -H[1] / N])

    H_opt = optimize.root(F_root, H_iv, method="lm", options={"xtol": 1.49012e-8})
    if not (H_opt.success):
        sys.exit("H optimization failed to converge.")

    H_hat = H_opt.x
    if params.verbose:
        print(f"LQ optima are: {H_hat}")
    return H_hat[0], H_hat[1]
