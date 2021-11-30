# From https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lqcontrol.py and https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/matrix_eqn.py

# Copyright © 2013-2021 Thomas J. Sargent and John Stachurski: BSD-3
# All rights reserved.

# Moved here to avoid numba dependency in the full qe python library
import numpy as np
from numpy import dot
from scipy.linalg import solve
from numpy.linalg import solve
from scipy.linalg import solve_discrete_lyapunov as sp_solve_discrete_lyapunov
from scipy.linalg import solve_discrete_are as sp_solve_discrete_are


def solve_discrete_riccati(A, B, Q, R, N=None, tolerance=1e-10, max_iter=500, method="doubling"):
    """
    Solves the discrete-time algebraic Riccati equation
    .. math::
        X = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q
    Computation is via a modified structured doubling algorithm, an
    explanation of which can be found in the reference below, if
    `method="doubling"` (default), and via a QZ decomposition method by
    calling `scipy.linalg.solve_discrete_are` if `method="qz"`.
    Parameters
    ----------
    A : array_like(float, ndim=2)
        k x k array.
    B : array_like(float, ndim=2)
        k x n array
    Q : array_like(float, ndim=2)
        k x k, should be symmetric and non-negative definite
    R : array_like(float, ndim=2)
        n x n, should be symmetric and positive definite
    N : array_like(float, ndim=2)
        n x k array
    tolerance : scalar(float), optional(default=1e-10)
        The tolerance level for convergence
    max_iter : scalar(int), optional(default=500)
        The maximum number of iterations allowed
    method : string, optional(default="doubling")
        Describes the solution method to use.  If it is "doubling" then
        uses the doubling algorithm to solve, if it is "qz" then it uses
        `scipy.linalg.solve_discrete_are` (in which case `tolerance` and
        `max_iter` are irrelevant).
    Returns
    -------
    X : array_like(float, ndim=2)
        The fixed point of the Riccati equation; a k x k array
        representing the approximate solution
    References
    ----------
    Chiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. "STRUCTURED DOUBLING
    ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR
    CONTROL WEIGHTING MATRICES." Taiwanese Journal of Mathematics 14, no. 3A
    (2010): pp-935.
    """
    EPS = np.finfo(float).eps
    methods = ["doubling", "qz"]
    if method not in methods:
        msg = "Check your method input. Should be {} or {}".format(*methods)
        raise ValueError(msg)

    # == Set up == #
    error = tolerance + 1
    fail_msg = "Convergence failed after {} iterations."

    # == Make sure that all array_likes are np arrays, two-dimensional == #
    A, B, Q, R = np.atleast_2d(A, B, Q, R)
    n, k = R.shape[0], Q.shape[0]
    I = np.identity(k)
    if N is None:
        N = np.zeros((n, k))
    else:
        N = np.atleast_2d(N)

    if method == "qz":
        X = sp_solve_discrete_are(A, B, Q, R, s=N.T)
        return X

    # if method == 'doubling'
    # == Choose optimal value of gamma in R_hat = R + gamma B'B == #
    current_min = np.inf
    candidates = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0, 10e5)
    BB = dot(B.T, B)
    BTA = dot(B.T, A)
    for gamma in candidates:
        Z = R + gamma * BB
        cn = np.linalg.cond(Z)
        if cn * EPS < 1:
            Q_tilde = -Q + dot(N.T, solve(Z, N + gamma * BTA)) + gamma * I
            G0 = dot(B, solve(Z, B.T))
            A0 = dot(I - gamma * G0, A) - dot(B, solve(Z, N))
            H0 = gamma * dot(A.T, A0) - Q_tilde
            f1 = np.linalg.cond(Z, np.inf)
            f2 = gamma * f1
            f3 = np.linalg.cond(I + dot(G0, H0))
            f_gamma = max(f1, f2, f3)
            if f_gamma < current_min:
                best_gamma = gamma
                current_min = f_gamma

    # == If no candidate successful then fail == #
    if current_min == np.inf:
        msg = "Unable to initialize routine due to ill conditioned arguments"
        raise ValueError(msg)

    gamma = best_gamma
    R_hat = R + gamma * BB

    # == Initial conditions == #
    Q_tilde = -Q + dot(N.T, solve(R_hat, N + gamma * BTA)) + gamma * I
    G0 = dot(B, solve(R_hat, B.T))
    A0 = dot(I - gamma * G0, A) - dot(B, solve(R_hat, N))
    H0 = gamma * dot(A.T, A0) - Q_tilde
    i = 1

    # == Main loop == #
    while error > tolerance:

        if i > max_iter:
            raise ValueError(fail_msg.format(i))

        else:
            A1 = dot(A0, solve(I + dot(G0, H0), A0))
            G1 = G0 + dot(dot(A0, G0), solve(I + dot(H0, G0), A0.T))
            H1 = H0 + dot(A0.T, solve(I + dot(H0, G0), dot(H0, A0)))

            error = np.max(np.abs(H1 - H0))
            A0 = A1
            G0 = G1
            H0 = H1
            i += 1

    return H1 + gamma * I  # Return X


class LQ:
    def __init__(self, Q, R, A, B, C=None, N=None, beta=1, T=None, Rf=None):
        # == Make sure all matrices can be treated as 2D arrays == #
        converter = lambda X: np.atleast_2d(np.asarray(X, dtype="float"))
        self.A, self.B, self.Q, self.R, self.N = list(map(converter, (A, B, Q, R, N)))
        # == Record dimensions == #
        self.k, self.n = self.Q.shape[0], self.R.shape[0]

        self.beta = beta

        if C is None:
            # == If C not given, then model is deterministic. Set C=0. == #
            self.j = 1
            self.C = np.zeros((self.n, self.j))
        else:
            self.C = converter(C)
            self.j = self.C.shape[1]

        if N is None:
            # == No cross product term in payoff. Set N=0. == #
            self.N = np.zeros((self.k, self.n))

        if T:
            # == Model is finite horizon == #
            self.T = T
            self.Rf = np.asarray(Rf, dtype="float")
            self.P = self.Rf
            self.d = 0
        else:
            self.P = None
            self.d = None
            self.T = None

            if (self.C != 0).any() and beta >= 1:
                raise ValueError(
                    "beta must be strictly smaller than 1 if " + "T = None and C != 0."
                )

        self.F = None

    def stationary_values(self, method="doubling"):
        # === simplify notation === #
        Q, R, A, B, N, C = self.Q, self.R, self.A, self.B, self.N, self.C

        # === solve Riccati equation, obtain P === #
        A0, B0 = np.sqrt(self.beta) * A, np.sqrt(self.beta) * B
        P = solve_discrete_riccati(A0, B0, R, Q, N, method=method)

        # == Compute F == #
        S1 = Q + self.beta * dot(B.T, dot(P, B))
        S2 = self.beta * dot(B.T, dot(P, A)) + N
        F = solve(S1, S2)

        # == Compute d == #
        if self.beta == 1:
            d = 0
        else:
            d = self.beta * np.trace(dot(P, dot(C, C.T))) / (1 - self.beta)

        # == Bind states and return values == #
        self.P, self.F, self.d = P, F, d

        return P, F, d