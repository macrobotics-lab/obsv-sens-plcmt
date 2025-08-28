"""
find_params.py

Calculates beam model parameters and then writes all parameters to params.yaml:
- Flexural rigidity (EI) and density (rho) of the beam.
- 'b', the coefficient that scales input pressure to a uniformly distributed external
force.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy as sp
import sympy as sym
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from beam import Beam

script_dir = os.path.dirname(os.path.abspath(__file__))

# A params.yaml file with the length of the beam and the number of modes must be present in the results directory.
yaml = YAML()

# Load the parameters from the params.yaml file
params_path = os.path.join(script_dir, "..", f"results/params.yaml")
with open(params_path, "r") as file:
    params = yaml.load(file)

L = params["beam"]["length"]  # Length of the beam [m]
N = params["beam"]["N"]  # Number of modes used in Rayleigh-Ritz method

# Load parameters from temp_params.json file obtained from the matlab scripts
temp_params_path = os.path.join(script_dir, "temp_params.json")
with open(temp_params_path, "r") as file:
    data = json.load(file)

nat_freqs = data["nat_freqs"][:N]  # Here, first 2 natural frequencies [rad/s]
zetas = data["zetas"]  # Damping ratios

# Initialize beam object with rho and EI set to 1.0
beam = Beam(L=L, EI=1, rho=1, N=N, zetas=zetas)

# Since EI and rho were set to 1, the M and K matrices are effectively scaled by EI and
# rho.
M_over_rho = beam.M
K_over_EI = beam.K

# Target eigenvalues (natural frequencies) for the beam model
lambda_desired = np.array(nat_freqs) ** 2

# Initial guess for the parameters (EI and rho), x0 = [EI_0, rho_0]
x0 = np.ones(N)


def cost_func_eigvals(x):
    """
    Cost function to minimize the difference between the desired and calculated
    eigenvalues.

    Args:
        x (array): Array containing the parameters [EI, rho].

    Returns:
        float: The cost value.
    """
    EI, rho = x
    M = M_over_rho * rho
    K = K_over_EI * EI
    eigvals, _ = sp.linalg.eigh(K, M)  # solve generalized eigenvalue problem
    return np.sum((eigvals - lambda_desired) ** 2)


# constraints
cons = (
    {"type": "ineq", "fun": lambda x: x[0]},  # EI >= 0
    {"type": "ineq", "fun": lambda x: x[1]},  # rho >= 0
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    result = sp.optimize.minimize(
        cost_func_eigvals, x0, constraints=cons, method="trust-constr"
    )

print("EI [Nm^2]:", result.x[0])
print("rho [kg/m]:", result.x[1], "\n")

eigvals, _ = sp.linalg.eigh(K_over_EI * result.x[0], M_over_rho * result.x[1])
print("Resulting natural frequencies [rad/s] with optimized EI and rho:")
print(np.sqrt(eigvals))

print("Target natural frequencies [rad/s]:")
print(nat_freqs, "\n")

# Save parameters to params.yaml
params["beam"]["EI"] = float(result.x[0])  # Flexural rigidity [Nm^2]
params["beam"]["rho"] = float(result.x[1])  # Density [kg/m]
params["beam"]["zetas"] = zetas  # Damping ratios

## Calculate the coefficient b that scales the input pressure to a uniformly distributed
# force
# Initialize beam object with optimized EI and rho
beam = Beam(L=L, EI=params["beam"]["EI"], rho=params["beam"]["rho"], N=N, zetas=zetas)


def get_measurements(p_set):
    """
    Load the measured points and compute the mean pressure applied to the beam during
    the step response. The measured points are the last recorded position of the 2D
    centroids.

    Args:
        p_set (int): Set pressure during the step response [kPa].

    Returns:
        np.ndarray: The measured points.
    """
    # Load the measured points from the CSV file
    df = pd.read_csv(os.path.join(script_dir, f"../../data/step/{p_set}kPa.csv"))

    # 2D position [mm] of all 7 centroids at the last recorded time step
    measured_points = np.zeros((2, 7))
    for i in range(7):
        measured_points[0, i] = df[f"Spine2D{i + 1}_x"].iloc[-1]
        measured_points[1, i] = df[f"Spine2D{i + 1}_y"].iloc[-1]

    p_measured = df["Measured_pressure"]
    # Mean pressure [kPa]; step applied at t=1s, taking index slightly later than 1s
    mean_pressure = np.mean(p_measured[110:])

    return measured_points, mean_pressure


def cost_func_rmse(x, mean_pressure, measured_points):
    """
    Cost function to minimize the RMSE between the measured and calculated shape of the
    beam.

    Args:
        x (array): Array containing the parameters [b].
        mean_pressure (float): Mean pressure applied to the beam in step input.
        measured_points (array): Measured shape of the beam.

    Returns:
        float: The cost value.
    """
    b = x[0]  # Coefficient to solve for
    f = mean_pressure * b  # External force applied to the beam
    u = beam.get_genforce_Q(f)  # Generalized force

    x_ss = np.linalg.solve(beam.A, beam.B @ u)  # Steady state solution
    z = x_ss[:N]
    q = beam.V @ np.linalg.inv(beam.Omega) @ z
    X = np.linspace(0, L, num=49)
    w = np.zeros(len(X))  # deflection of the beam at points X [m]
    for i in range(len(X)):
        w[i] = np.sum(
            [
                q[j] * beam.basis_functions[j].subs(sym.symbols("x"), X[i])
                for j in range(N)
            ]
        )

    # Get the computed shape of the beam at the same location as the measured points
    computed_x = -w[0::8] * 1000  # Convert to mm
    computed_y = -X[0::8] * 1000  # Convert to mm

    rmse = np.sqrt(
        np.mean(
            (computed_x - measured_points[0, :]) ** 2
            + (computed_y - measured_points[1, :]) ** 2
        )
    )
    return rmse


b_list = []  # List to store the b values for each set pressure
# Start at p_set = 15 kPa since lower values do not activate the valves
for p_set in range(15, 101, 5):
    # Get the measured points and mean pressure for the current set pressure
    measured_points, mean_pressure = get_measurements(p_set)

    # Initial guess for b
    x0 = np.array([1])

    # Optimize b using the cost function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = sp.optimize.minimize(
            cost_func_rmse,
            x0,
            args=(mean_pressure, measured_points),
            method="trust-constr",
        )

    # print(f"b for {p_set} kPa:", result.x[0])
    b_list.append(result.x[0])

print("Mean coefficient b scaling pressure to force:")
print(np.mean(np.array(b_list)))  # Mean value of b over all set pressures

# Save the b value to params.yaml
params["beam"]["b"] = float(np.mean(np.array(b_list)))  # Coefficient b

with open(params_path, "w") as file:
    yaml.dump(params, file)
