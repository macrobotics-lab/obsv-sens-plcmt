"""
beam.py
"""

import numpy as np
import scipy as sp
import sympy as sym
from ruamel.yaml import YAML


class Beam:
    """
    Class representing an Euler-Bernoulli beam.
    """

    def __init__(self, L, EI, rho, N, zetas):
        """
        Initialize a Beam object.

        Args:
            L (float): Length of the beam [m].
            EI (float): Flexural rigidity (Young's modulus * moment of inertia) [Nm^2].
            rho (float): Mass per unit length [kg/m].
            N (int): Number of modes used in Rayleigh-Ritz method.
            zetas (list or array-like): Damping ratios for each mode
                (must be of length N).
        """
        self.L = L
        self.EI = EI
        self.rho = rho
        self.N = N
        self.zetas = zetas
        self.b = None  # Parameter for pressure to force conversion, from params.yaml
        self.Q_base = None  # Base generalized force vector, computed once and reused

        self._get_basis_funcs()
        self.M, self.K = self._get_M_and_K()
        self.V, self.Omega = self._get_V_and_Omega()
        self.A, self.B = self._get_A_and_B()

        self.basis_functions_numeric = [
            sym.lambdify(sym.symbols("x"), func) for func in self.basis_functions
        ]

    def _get_basis_funcs(self):
        """
        Create the basis functions for the beam. The basis functions are the
        mode shapes of a cantilever beam.

        Returns:
            list: List of symbolic expressions representing the basis functions.
        """
        basis_functions = []
        x = sym.symbols("x")
        x_ini = (0.5 + np.arange(self.N)) * np.pi
        beta_Ls = sp.optimize.fsolve(lambda x: np.cosh(x) * np.cos(x) + 1, x_ini)
        for beta_L in beta_Ls:
            basis_functions.append(
                sym.sin(beta_L * x / self.L)
                - sym.sinh(beta_L * x / self.L)
                - (np.sin(beta_L) + np.sinh(beta_L))
                / (np.cos(beta_L) + np.cosh(beta_L))
                * (sym.cos(beta_L * x / self.L) - sym.cosh(beta_L * x / self.L))
            )
        self.basis_functions = basis_functions

    def _get_M_and_K(self):
        """
        Compute the mass and stiffness matrices for the beam.

        Returns:
            tuple: Mass matrix (M) and stiffness matrix (K).
        """
        x = sym.symbols("x")
        M = self.rho * sym.integrate(
            sym.Matrix([self.basis_functions]).T * sym.Matrix([self.basis_functions]),
            (x, 0, self.L),
        )
        K = self.EI * sym.integrate(
            sym.diff(sym.Matrix([self.basis_functions]).T, x, 2)
            * sym.diff(sym.Matrix([self.basis_functions]), x, 2),
            (x, 0, self.L),
        )
        return np.array(M).astype(np.float64), np.array(K).astype(np.float64)

    def _get_V_and_Omega(self):
        """
        Compute the mass-normalized eigenvector matrix V and the natural frequency
        matrix Omega.

        Returns:
            tuple: Matrices V and Omega.
        """
        D, Q = np.linalg.eig(np.linalg.inv(self.M) @ self.K)
        V = Q / np.sqrt(np.diag(Q.T @ self.M @ Q))
        Omega = np.diag(np.sqrt(D))
        return V, Omega

    def get_genforce_Q(self, f):
        """
        Compute the generalized force vector Q.

        Args:
            f (array-like): External force vector [N/m].

        Returns:
            array: Generalized force vector Q.
        """
        x = sym.symbols("x")
        Q = sym.integrate(sym.Matrix([self.basis_functions]) * f, (x, 0, self.L))
        return np.array(Q.T).astype(np.float64)

    def get_genforce_from_pressure(self, p_input):
        """
        Compute the generalized force vector Q from input pressure P, which is
        approximated as a uniformly distributed force across the beam.

        Args:
            p_input (float): Input pressure [kPa].

        Returns:
            array: Generalized force vector Q.
        """
        # Since only a constant (p_input) is changing the force at each function call,
        # first compute the integral with f = 1, then multiply by p_input to improve
        # computational efficiency.

        # check if optimized parameter b from params.yaml has been loaded
        if self.b is None:
            yaml = YAML()
            with open(f"results/params.yaml", "r") as file:
                params = yaml.load(file)
                # parameter b for pressure to force conversion
                self.b = params["beam"]["b"]

        # check if Q_base has been computed
        if self.Q_base is None:
            # Compute the base generalized force vector Q_base
            self.Q_base = self.get_genforce_Q(1)

        # Compute the generalized force vector Q from pressure
        Q = self.Q_base * self.b * p_input
        return Q

    def _get_A_and_B(self):
        """
        Compute the dynamics matrix A and the input matrix B.

        Returns:
            tuple: Matrices A and B.
        """
        zeta = np.array(self.zetas)
        A = np.block(
            [
                [np.zeros((self.N, self.N)), self.Omega],
                [-self.Omega, -2 * np.diag(zeta) @ self.Omega],
            ]
        )
        B = np.block([[np.zeros((self.N, self.N))], [self.V.T]])
        return A, B

    def range_measurement_matrix(self, X):
        """
        Computes the range measurement matrix C. Each row of C corresponds to the
        range measurement taken at the location specified by X[i].

        Args:
            X (array-like): Locations where the range is measured [m].

        Returns:
            np.ndarray: Range measurement matrix C.
        """
        # Compute Phi
        Phi = np.zeros((len(X), self.N))
        x = sym.symbols("x")
        for i, s in enumerate(X):
            for j, basis_func in enumerate(self.basis_functions):
                Phi[i, j] = basis_func.subs(x, s)
        C = np.zeros((len(X), Phi.shape[1] * 2))
        for i, phi_row in enumerate(Phi):
            C[i, : self.N] = phi_row @ self.V @ np.linalg.inv(self.Omega)
        C[:, self.N :] = np.zeros((len(X), self.N))
        return C

    def strain_measurement_matrix(self, X):
        """
        Computes the strain measurement matrix C. Each row of C corresponds to the
        strain measurement taken at the location specified by X[i].

        Args:
            X (array-like): Locations where the strain is measured [m].

        Returns:
            np.ndarray: Strain measurement matrix C.
        """
        r = 2e-2  # [m] average radius of beam, strain computed at 2cm from neutral axis
        # Compute d^2/dx^2 (Phi)
        dPhi = np.zeros((len(X), self.N))
        x = sym.symbols("x")
        for i, s in enumerate(X):
            for j, basis_func in enumerate(self.basis_functions):
                second_diff = sym.diff(basis_func, x, 2)
                dPhi[i, j] = second_diff.subs(x, s)
        C = np.zeros((len(X), dPhi.shape[1] * 2))
        for i, dphi_row in enumerate(dPhi):
            C[i, : self.N] = r * dphi_row @ self.V @ np.linalg.inv(self.Omega)
        C[:, self.N :] = np.zeros((len(X), self.N))
        return C
