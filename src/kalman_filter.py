"""
kalman_filter.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import scipy as sp
import sympy as sym
from ruamel.yaml import YAML

from beam import Beam


def get_pressure_var(folder_path):
    """
    Get the variance of the measured input pressure.

    Args:
        folder_path (str): Path to the folder containing the step response data.

    Returns:
        float: Variance of the pressure measurement.
    """
    vars = []
    for p_set in range(15, 101, 5):
        # Read the CSV file for the given pressure set
        df_name = f"{folder_path}/{p_set}kPa.csv"
        data = pd.read_csv(df_name)
        time = data["Time"]
        p_meas = data["Measured_pressure"]

        # Step input applied at t = 1s
        vars.append(np.var(p_meas[time > 1.1]))

    # Return the average variance of the pressure measurements
    return np.mean(vars)


def get_pos_var(error):
    """
    Get the variance of the position measurement error. The calibration error is taken as the standard deviation of the
    noise.

    Args:
        error (float): Reported calibration error [m].

    Returns:
        float: Variance of the position measurement error.
    """
    return error**2


def discretize(A_cont, B_cont, C_cont, L_cont, M_cont, dt):
    """
    Continous-time to discrete-time conversion of state-space matrices using Van Loan's method.

    Args:
        A_cont (np.ndarray): Continuous-time state matrix.
        B_cont (np.ndarray): Continuous-time input matrix.
        C_cont (np.ndarray): Continuous-time output matrix.
        L_cont (np.ndarray): Continuous-time disturbance matrix.
        M_cont (np.ndarray): Continuous-time noise matrix.
        dt (float): Sampling time step.

    Returns:
        tuple: Discretized matrices (A_d, B_d, C_d, L_d, M_d).
    """
    Xi = np.block(
        [
            [A_cont, B_cont],
            [np.zeros((B_cont.shape[1], A_cont.shape[0])), np.eye(B_cont.shape[1])],
        ]
    )
    Ups = sp.linalg.expm(Xi * dt)

    A_d = Ups[: A_cont.shape[0], : A_cont.shape[1]]
    B_d = Ups[: A_cont.shape[0], A_cont.shape[1] :]

    L_d = L_cont
    C_d = C_cont
    M_d = M_cont
    return A_d, B_d, C_d, L_d, M_d


def approximate_strain_measurements(beam, pos_candidate_locations, strain_candidate_locations, dataframe):
    """
    Approximate the strain measurements at candidate sensor locations using the measured position data.

    Args:
        beam (Beam): Beam object.
        pos_candidate_locations (list): Candidate locations for the position sensors (where the measurements were taken)
        strain_candidate_locations (list): Candidate locations for the strain sensors
        (where the measurements are to be approximated).
        dataframe (pd.DataFrame): DataFrame containing the measured data.

    Returns:
        np.ndarray: Approximated strain values at the sensor locations.
    """
    # Get position measurements from dataframe
    time = dataframe["Time"]
    last_col_name = dataframe.columns[-1]
    num_centroids = int(last_col_name.strip("Spine2D").split("_")[0])
    spine2D = np.zeros((len(time), 2, num_centroids))
    for i in range(num_centroids):
        spine2D[:, 0, i] = dataframe[f"Spine2D{i + 1}_x"] * 1e-3  # Convert mm to m
        spine2D[:, 1, i] = dataframe[f"Spine2D{i + 1}_y"] * 1e-3  # Convert mm to m

    # Get second derivative of the basis functions
    diff2_basis_funcs = [sym.diff(basis_func, sym.symbols("x"), 2) for basis_func in beam.basis_functions]
    diff2_basis_funcs_numeric = [sym.lambdify(sym.symbols("x"), func) for func in diff2_basis_funcs]

    # Compute strain measurements by first solving for q with least squares
    A_lstsq = np.zeros((num_centroids, beam.N))
    for i, s in zip(range(num_centroids), pos_candidate_locations):
        A_lstsq[i, :] = [
            beam.basis_functions[0].subs(sym.symbols("x"), s),
            beam.basis_functions[1].subs(sym.symbols("x"), s),
        ]

    y_strain = np.zeros((len(strain_candidate_locations), len(time)))
    for t in range(len(time)):
        q, _, _, _ = np.linalg.lstsq(A_lstsq, -spine2D[t, 0, :].reshape(-1, 1), rcond=None)
        y_strain_approx = np.zeros((len(strain_candidate_locations), 1))
        for j, loc in enumerate(strain_candidate_locations):
            for i in range(beam.N):
                y_strain_approx[j] += q[i, 0] * diff2_basis_funcs_numeric[i](loc) * 2e-2  # 2 cm from neutral axis
        y_strain[:, t] = y_strain_approx.flatten()

    return y_strain


def run_kf(beam, sensor_type, candidate_locations, opt_indices, y, u, Q_var, R_var, dt, t_eval):
    """
    Run the Kalman filter on the given data with the specified sensor type and optimized placement.

    Args:
        beam (Beam): Beam object.
        sensor_type (str): Type of sensor ("Position" or "Strain" or "Both").
        candidate_locations (list or tuple of lists): Candidate locations for the sensors.
        opt_indices (np.ndarray or tuple of np.ndarrays): Indices of the optimized sensor placements.
        y (np.ndarray): len(opt_indices) x len(time) matrix of measurements.
        u (np.ndarray): Input pressure measurements.
        Q_var (float): Variance of the process noise (pressure).
        R_var (float or tuple of floats): Variance of the measurement noise (position or/and strain).
        dt (float): Sampling time step.
        t_eval (float): Final time step for the Kalman filter.

    Returns:
        x_hat (np.ndarray): Estimated state vector after applying the Kalman filter.
    """
    # Get the continuous-time state-space matrices
    A_cont, B_cont = beam.A, beam.B

    # Get the measurement matrix for the selected sensor type
    if sensor_type == "Position":
        C_cont = beam.range_measurement_matrix([candidate_locations[i] for i in opt_indices])
    elif sensor_type == "Strain":
        C_cont = beam.strain_measurement_matrix([candidate_locations[i] for i in opt_indices])
    elif sensor_type == "Both":
        C_cont_pos = beam.range_measurement_matrix([candidate_locations[0][i] for i in opt_indices[0]])
        C_cont_strain = beam.strain_measurement_matrix([candidate_locations[1][i] for i in opt_indices[1]])
        C_cont = np.vstack((C_cont_pos, C_cont_strain))

    L_cont = np.block(
        [
            [np.zeros((beam.N, 1))],
            [np.ones((beam.N, 1))],
        ]
    )
    M_cont = np.eye(C_cont.shape[0])

    # Discretize the continuous-time matrices
    A_d, B_d, C_d, L_d, M_d = discretize(A_cont, B_cont, C_cont, L_cont, M_cont, dt)

    # Covariance matrices
    Q_k = np.array([[Q_var]])  # Process noise covariance
    # Measurement noise covariance
    if sensor_type == "Both":
        R_k = np.diag([R_var[0]] * len(opt_indices[0]) + [R_var[1]] * len(opt_indices[1]))
    else:
        R_k = np.diag(np.ones(C_cont.shape[0])) * R_var

    # Initialize x_hat and P_hat to store them at each time step
    x_hat = np.zeros((beam.N * 2, int(t_eval / dt + 1)))
    P_hat = np.zeros((beam.N * 2, beam.N * 2, int(t_eval / dt + 1)))

    # Initial conditions
    x_hat[:, 0] = np.zeros(beam.N * 2)
    P_hat[:, :, 0] = np.eye(beam.N * 2) * 1e-1

    # Kalman filter loop
    for k in range(1, int(t_eval / dt) + 1):
        # Prediction step
        x_check = A_d @ x_hat[:, [k - 1]] + B_d @ beam.get_genforce_from_pressure(u[k - 1])
        P_check = A_d @ P_hat[:, :, k - 1] @ A_d.T + L_d @ Q_k @ L_d.T

        # Update step
        S_k = C_d @ P_check @ C_d.T + M_d @ R_k @ M_d.T
        K_k = P_check @ C_d.T @ np.linalg.inv(S_k)
        x_hat[:, [k]] = x_check + K_k @ (y[:, [k]] - C_d @ x_check)
        P_hat[:, :, k] = (np.eye(beam.N * 2) - K_k @ C_d) @ P_check

    return x_hat


def get_w_from_x(x, beam):
    """
    Get deflection of the beam from the state vector x.

    Args:
        x (np.ndarray): State vector of the beam at all time steps.
        beam (Beam): Beam object.

    Returns:
        tuple:
            X (np.ndarray): Points along the beam where deflection is computed.
            w (np.ndarray): Deflection of the beam at points along the beam.
    """
    z = x[: beam.N, :]
    q = beam.V @ np.linalg.inv(beam.Omega) @ z
    X = np.linspace(0, beam.L, int(beam.L * 1000 + 1))
    # Precompute all basis functions at all X
    basis_matrix = np.array([[f(x_i) for f in beam.basis_functions_numeric] for x_i in X])  # shape: (100, N)
    w = basis_matrix @ q  # shape: (100, num_time_steps)
    return X, w


def get_rmse(x_est, x_gt, pos_candidate_locations):
    """
    Calculate the root mean square error (RMSE) between the estimated state vector x
    and the ground truth state vector x_gt.

    Args:
        x_est (np.ndarray): Estimated state vector.
        x_gt (np.ndarray): Ground truth state vector.
        pos_candidate_locations (list): Candidate locations for position sensors, where
        the RMSE is computed.

    Returns:
        tuple:
            rmse (np.ndarray): RMSE for each time step.
            rmse_traj (float): Overall trajectory RMSE.
    """
    X, w_est = get_w_from_x(x_est, beam)
    _, w_gt = get_w_from_x(x_gt, beam)

    indices = np.where(np.isin(X, pos_candidate_locations))[0]
    x_coord_est = -w_est[indices, :] * 1e3  # Convert m to mm
    y_coord_est = -X[indices] * 1e3  # Convert m to mm
    x_coord_gt = -w_gt[indices, :] * 1e3  # Convert m to mm
    y_coord_gt = -X[indices] * 1e3  # Convert m to mm

    big_x_est = np.vstack((x_coord_est, np.tile(y_coord_est, (w_est.shape[1], 1)).T))
    big_x_gt = np.vstack((x_coord_gt, np.tile(y_coord_gt, (w_gt.shape[1], 1)).T))

    rmse = np.sqrt((1 / np.shape(big_x_est)[0]) * np.linalg.norm(big_x_gt - big_x_est, axis=0) ** 2)
    rmse_traj = np.sqrt(np.mean(np.linalg.norm(big_x_gt - big_x_est, axis=0) ** 2))

    return rmse, rmse_traj, big_x_est, big_x_gt


def run_kf_on_data(
    beam,
    spr,
    pos_candidate_locations,
    strain_candidate_locations,
    result_key_prefix,
    result_dict,
    pressure_var,
    pos_var,
    strain_var,
    dt,
    t_eval,
):
    """
    Run the Kalman filter on the data for both position and strain sensors,
    and store the results in the result_dict.

    Args:
        beam (Beam): Beam object.
        spr (dict): Sensor placement results dictionary, loaded from YAML.
        pos_candidate_locations (list): Candidate locations for position sensors.
        strain_candidate_locations (list): Candidate locations for strain sensors.
        result_key_prefix (str): Prefix for the result keys in the result_dict.
        result_dict (dict): Dictionary to store the results.
        pressure_var (float): Variance of the pressure measurement.
        pos_var (float): Variance of the position measurement.
        strain_var (float): Variance of the strain measurement.
        dt (float): Sampling time step.
        t_eval (float): Final time step.

    Returns:
        None: The results are stored in the result_dict.
    """

    time = df["Time"]
    p_meas = df["Measured_pressure"]

    spine2D = np.zeros((len(time), 2, len(pos_candidate_locations)))
    for i in range(len(pos_candidate_locations)):
        spine2D[:, 0, i] = df[f"Spine2D{i + 1}_x"]
        spine2D[:, 1, i] = df[f"Spine2D{i + 1}_y"]

    # generate all strain measurements
    y_strain_all = approximate_strain_measurements(beam, pos_candidate_locations, strain_candidate_locations, df)

    # ground truth results
    y = np.zeros((len(pos_candidate_locations), len(time)))
    for i in range(len(pos_candidate_locations)):
        y[i, :] = -spine2D[:, 0, i] * 1e-3  # Convert mm to m
    x_hat_gt = run_kf(
        beam,
        "Position",
        pos_candidate_locations,
        np.arange(len(pos_candidate_locations)),
        y,
        p_meas,
        pressure_var,
        pos_var,
        dt,
        t_eval,
    )
    result_dict[result_key_prefix + ("Ground Truth",)] = {"x_hat": x_hat_gt}

    # Sensor placement results
    for p in [1, 2, 3]:
        # optimized position sensor placement results
        for obj_func, result in spr["Position"][f"p = {p}"].items():
            if "Optimal Indices" in result:
                opt_indices_pos = result["Optimal Indices"]
                y_pos = np.zeros((len(opt_indices_pos), len(time)))
                for i, idx in enumerate(opt_indices_pos):
                    y_pos[i, :] = -spine2D[:, 0, idx] * 1e-3  # Convert mm to m
                x_hat_pos = run_kf(
                    beam,
                    "Position",
                    pos_candidate_locations,
                    opt_indices_pos,
                    y_pos,
                    p_meas,
                    pressure_var,
                    pos_var,
                    dt,
                    t_eval,
                )
                rmse_pos, rmse_traj_pos, est_coords_pos, gt_coords_pos = get_rmse(
                    x_hat_pos, x_hat_gt, pos_candidate_locations
                )
                result_dict[result_key_prefix + ("Position", f"p = {p}", obj_func.split()[-1])] = {
                    "x_hat": x_hat_pos,
                    "rmse": rmse_pos,
                    "rmse_traj": rmse_traj_pos,
                    "est_coords": est_coords_pos,
                    "gt_coords": gt_coords_pos,
                }

        # optimized strain sensor placement results
        for obj_func, result in spr["Strain"][f"p = {p}"].items():
            if "Optimal Indices" in result:
                opt_indices_strain = result["Optimal Indices"]
                y_strain = y_strain_all[opt_indices_strain, :]
                x_hat_strain = run_kf(
                    beam,
                    "Strain",
                    strain_candidate_locations,
                    opt_indices_strain,
                    y_strain,
                    p_meas,
                    pressure_var,
                    strain_var,
                    dt,
                    t_eval,
                )
                rmse_strain, rmse_traj_strain, est_coords_strain, gt_coords_strain = get_rmse(
                    x_hat_strain, x_hat_gt, pos_candidate_locations
                )
                result_dict[result_key_prefix + ("Strain", f"p = {p}", obj_func.split()[-1])] = {
                    "x_hat": x_hat_strain,
                    "rmse": rmse_strain,
                    "rmse_traj": rmse_traj_strain,
                    "est_coords": est_coords_strain,
                    "gt_coords": gt_coords_strain,
                }

        # position baseline results
        bas_indices_pos = spr["Position"][f"p = {p}"]["Baseline Indices"]
        y_pos = np.zeros((len(bas_indices_pos), len(time)))
        for i, idx in enumerate(bas_indices_pos):
            y_pos[i, :] = -spine2D[:, 0, idx] * 1e-3  # Convert mm to m
        x_hat_pos = run_kf(
            beam,
            "Position",
            pos_candidate_locations,
            bas_indices_pos,
            y_pos,
            p_meas,
            pressure_var,
            pos_var,
            dt,
            t_eval,
        )
        rmse_pos, rmse_traj_pos, est_coords_pos, gt_coords_pos = get_rmse(x_hat_pos, x_hat_gt, pos_candidate_locations)
        result_dict[result_key_prefix + ("Position", f"p = {p}", "Baseline")] = {
            "x_hat": x_hat_pos,
            "rmse": rmse_pos,
            "rmse_traj": rmse_traj_pos,
            "est_coords": est_coords_pos,
            "gt_coords": gt_coords_pos,
        }

        # strain baseline results
        bas_indices_strain = spr["Strain"][f"p = {p}"]["Baseline Indices"]
        y_strain = y_strain_all[bas_indices_strain, :]
        x_hat_strain = run_kf(
            beam,
            "Strain",
            strain_candidate_locations,
            bas_indices_strain,
            y_strain,
            p_meas,
            pressure_var,
            strain_var,
            dt,
            t_eval,
        )
        rmse_strain, rmse_traj_strain, est_coords_strain, gt_coords_strain = get_rmse(
            x_hat_strain, x_hat_gt, pos_candidate_locations
        )
        result_dict[result_key_prefix + ("Strain", f"p = {p}", "Baseline")] = {
            "x_hat": x_hat_strain,
            "rmse": rmse_strain,
            "rmse_traj": rmse_traj_strain,
            "est_coords": est_coords_strain,
            "gt_coords": gt_coords_strain,
        }

    # Results for combined position and strain sensors
    for p1p2_pair in [key for key in spr["Both"].keys() if "p1" in key]:
        for obj_func, result in spr["Both"][p1p2_pair].items():
            if "Optimal Indices Position" in result and "Optimal Indices Strain" in result:
                opt_indices_pos = result["Optimal Indices Position"]
                opt_indices_strain = result["Optimal Indices Strain"]

                y_pos = np.zeros((len(opt_indices_pos), len(time)))
                for i, idx in enumerate(opt_indices_pos):
                    y_pos[i, :] = -spine2D[:, 0, idx] * 1e-3  # Convert mm to m
                y_strain = y_strain_all[opt_indices_strain, :]
                y = np.vstack((y_pos, y_strain))
                x_hat_both = run_kf(
                    beam,
                    "Both",
                    (pos_candidate_locations, strain_candidate_locations),
                    (opt_indices_pos, opt_indices_strain),
                    y,
                    p_meas,
                    pressure_var,
                    (pos_var, strain_var),
                    dt,
                    t_eval,
                )
                rmse_both, rmse_traj_both, est_coords_both, gt_coords_both = get_rmse(
                    x_hat_both, x_hat_gt, pos_candidate_locations
                )
                result_dict[result_key_prefix + ("Both", p1p2_pair, obj_func.split()[-1])] = {
                    "x_hat": x_hat_both,
                    "rmse": rmse_both,
                    "rmse_traj": rmse_traj_both,
                    "est_coords": est_coords_both,
                    "gt_coords": gt_coords_both,
                }

        # Add baseline results for combined sensors
        bas_indices_pos = spr["Both"][p1p2_pair]["Baseline Indices Position"]
        bas_indices_strain = spr["Both"][p1p2_pair]["Baseline Indices Strain"]
        y_pos = np.zeros((len(bas_indices_pos), len(time)))
        for i, idx in enumerate(bas_indices_pos):
            y_pos[i, :] = -spine2D[:, 0, idx] * 1e-3  # Convert mm to m
        y_strain = y_strain_all[bas_indices_strain, :]
        y = np.vstack((y_pos, y_strain))
        x_hat_both = run_kf(
            beam,
            "Both",
            (pos_candidate_locations, strain_candidate_locations),
            (bas_indices_pos, bas_indices_strain),
            y,
            p_meas,
            pressure_var,
            (pos_var, strain_var),
            dt,
            t_eval,
        )
        rmse_both, rmse_traj_both, est_coords_both, gt_coords_both = get_rmse(
            x_hat_both, x_hat_gt, pos_candidate_locations
        )
        result_dict[result_key_prefix + ("Both", p1p2_pair, "Baseline")] = {
            "x_hat": x_hat_both,
            "rmse": rmse_both,
            "rmse_traj": rmse_traj_both,
            "est_coords": est_coords_both,
            "gt_coords": gt_coords_both,
        }


# load target directories
yaml = YAML()
data_dir = "../data"

pressure_var = get_pressure_var(f"{data_dir}/step")
pos_var = get_pos_var(0.005)  # 0.5 mm calibration error inflated by an order of magnitude
strain_var = pos_var * 1e2  # pos_var inflated by two orders of magnitude

dt = 0.01  # Sampling time step [s]
t_eval = 10  # Final time step

# Load parameters from params.yaml
with open(f"results/params.yaml", "r") as file:
    params = yaml.load(file)
L = params["beam"]["length"]  # Length of the beam [m]
N = params["beam"]["N"]  # Number of modes used in Rayleigh-Ritz method
EI = params["beam"]["EI"]  # Flexural rigidity [Nm^2]
rho = params["beam"]["rho"]  # Mass per unit length [kg/m]
zetas = params["beam"]["zetas"]  # Damping ratios

beam = Beam(L, EI, rho, N, zetas)

with open(f"results/sensor_placement_results.yaml", "r") as file:
    spr = yaml.load(file)
pos_candidate_locations = spr["Position"]["Candidate Locations"]
strain_candidate_locations = spr["Strain"]["Candidate Locations"]

step_kf_results = {}
# Run Kalman filter on step response data
for p_input in np.arange(0, 101, 5):
    df = pd.read_csv(f"{data_dir}/step/{p_input}kPa.csv")

    run_kf_on_data(
        beam,
        spr,
        pos_candidate_locations,
        strain_candidate_locations,
        (f"{p_input}kPa",),
        step_kf_results,
        pressure_var,
        pos_var,
        strain_var,
        dt,
        t_eval,
    )

# Save results
os.makedirs(f"results/kf_results", exist_ok=True)
with open(f"results/kf_results/step_kf_results.pkl", "wb") as f:
    pickle.dump(step_kf_results, f)

sine_kf_results = {}
# Run Kalman filter on sine response data
for p_input_A in np.arange(10, (101 + 1) // 2, 10):
    for p_input_w in list(range(1, 7)):
        df = pd.read_csv(f"{data_dir}/sine/A{p_input_A}kPa_w{p_input_w}.csv")

        run_kf_on_data(
            beam,
            spr,
            pos_candidate_locations,
            strain_candidate_locations,
            (f"A{p_input_A}kPa_w{p_input_w}",),
            sine_kf_results,
            pressure_var,
            pos_var,
            strain_var,
            dt,
            t_eval,
        )

# Save results
with open(f"results/kf_results/sine_kf_results.pkl", "wb") as f:
    pickle.dump(sine_kf_results, f)
