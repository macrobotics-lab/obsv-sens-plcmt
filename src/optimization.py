"""
optimization.py

This script finds the optimal placement of position and strain sensors on a beam and
saves the results to a yaml file.
"""

import control as ct
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from beam import Beam

# Use style sheet for plotting
plt.style.use("my_default.mplstyle")


def optimize_placement(A, C1, p1, objective, C2=None, p2=None):
    """
    Optimize sensor placement for one of two sensing types.

    Args:
        A (np.ndarray): Dynamics matrix.
        C1 (np.ndarray): Measurement matrix for the first sensing type.
        p1 (int): Number of sensors for the first sensing type.
        objective (str): Objective function ("lambda_min", "trace", "log_det")
        C2 (np.ndarray, optional): Measurement matrix for the second sensing type.
        p2 (int, optional): Number of sensors for the second sensing type.

    Returns:
        tuple: (
            np.ndarray: Selector vector for the first sensing type.
            np.ndarray, optional: Selector vector for the second sensing type.
            float: Objective function value.
        )
    """
    # Initialize observability Gramian Wo
    Wo = cp.Variable((A.shape[0], A.shape[0]), symmetric=True)

    # Add first constraint enforcing Wo to be positive definite
    constraints = [1e-8 * np.eye(A.shape[0]) << Wo]

    # First sensing type
    alpha1 = cp.Variable((C1.shape[0], 1))
    alphaC1_sum = np.zeros((C1.shape[1], C1.shape[1]))
    for i in range(C1.shape[0]):
        alphaC1_sum = alphaC1_sum + alpha1[i, 0] * C1[[i], :].T @ C1[[i], :]
    constraints += [cp.sum(alpha1) == p1, alpha1 >= 0, alpha1 <= 1]

    # Second sensing type (optional)
    if C2 is not None and p2 is not None:
        alpha2 = cp.Variable((C2.shape[0], 1))
        alphaC2_sum = np.zeros((C2.shape[1], C2.shape[1]))
        for i in range(C2.shape[0]):
            alphaC2_sum = alphaC2_sum + alpha2[i, 0] * C2[[i], :].T @ C2[[i], :]
        constraints += [cp.sum(alpha2) == p2, alpha2 >= 0, alpha2 <= 1]
    else:
        alphaC2_sum = np.zeros((C1.shape[1], C1.shape[1]))

    # Lyapunov constraint
    constraints += [A.T @ Wo + Wo @ A + alphaC1_sum + alphaC2_sum == np.zeros((A.shape[0], A.shape[0]))]

    # Objective
    if objective == "lambda_min":
        obj = cp.Maximize(cp.lambda_min(Wo))
    elif objective == "trace":
        obj = cp.Maximize(cp.trace(Wo))
    elif objective == "log_det":
        obj = cp.Maximize(cp.log_det(Wo))
    else:
        raise ValueError("Invalid objective function.")

    prob = cp.Problem(obj, constraints)
    # prob.solve(solver=cp.MOSEK, verbose=True)
    try:
        prob.solve(
            solver=cp.MOSEK,
            verbose=False,
            mosek_params={"MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-3},
        )
    except cp.error.SolverError as e:
        if C2 is not None:
            return None, None, None
        else:
            return None, None

    if C2 is not None and p2 is not None:
        return alpha1.value, alpha2.value, prob.value
    else:
        return alpha1.value, prob.value


def is_binary(vector, tol=0.15):
    """
    Check if a vector is binary (0 or 1) within a tolerance.

    Args:
        vector (np.ndarray): The vector to check.
        tol (float): Tolerance for binary check.

    Returns:
        bool: True if the vector is binary, False otherwise.
    """
    return np.all(np.abs(vector - np.round(vector)) < tol)


# Compute optimal sensor placement for the beam
# Load parameters from params.yaml
yaml = YAML()

with open(f"results/params.yaml", "r") as file:
    params = yaml.load(file)
L = params["beam"]["length"]  # Length of the beam [m]
N = params["beam"]["N"]  # Number of modes used in Rayleigh-Ritz method
EI = params["beam"]["EI"]  # Flexural rigidity [Nm^2]
rho = params["beam"]["rho"]  # Mass per unit length [kg/m]
zetas = params["beam"]["zetas"]  # Damping ratios

beam = Beam(L, EI, rho, N, zetas)


def compute_gramian_metric(A, C_cand, idx, metric):
    """
    Compute the Gramian metric for a given sensor placement.

    Args:
        A (np.ndarray): Dynamics matrix.
        C_cand (np.ndarray): Measurement matrix with all candidate sensor locations.
        idx (list): Indices of the selected sensor placement.
        metric (str): Metric to compute ("lambda_min", "trace", "log_det").
    """
    G = ct.ss(A, np.zeros((A.shape[0], 1)), C_cand[idx, :], np.zeros((len(idx), 1)))
    Wo = ct.gram(G, "o")
    if metric == "lambda_min":
        return np.min(np.linalg.eigvals(Wo))
    elif metric == "trace":
        return np.trace(Wo)
    elif metric == "log_det":
        if np.linalg.det(Wo) <= 0:
            return -np.inf
        return np.log(np.linalg.det(Wo))


def baseline_indices(sensor_type, cand_locs, p):
    """
    Get baseline indices for sensor placement based on the sensor type and number of sensors.

    Args:
        sensor_type (str): Type of sensor ("Position" or "Strain").
        cand_locs (list): Candidate locations for the sensors.
        p (int): Number of sensors to place.

    Returns:
        list: Indices of the baseline sensor placement.
    """
    n = len(cand_locs)
    if sensor_type == "Position":
        if p == 1:
            return [n // 2]
        elif p == 2:
            return [n // 2, n - 1]
        elif p == 3:
            return [n // 3, 2 * n // 3, n - 1]
        else:
            # Generalize to equally spaced indices
            return [round(j * (n - 1) / (p - 1)) for j in range(p)]
    elif sensor_type == "Strain":
        if p == 1:
            return [0]
        elif p == 2:
            return [0, n // 2]
        elif p == 3:
            return [0, n // 3, 2 * n // 3]
        else:
            # Generalize to equally spaced indices
            return [round(j * (n - 1) / (p - 1)) for j in range(p)]


def process_optimal_placement(sensor_type, cand_locs, beam, p1p2pairs=None):
    """
    Find optimal sensor placement for a given sensor type and candidate locations, then save it to a dictionary for
    dumping to yaml file. Also add baseline placement.
    Args:
        sensor_type (str): Type of sensor ("Position" or "Strain" or "Both").
        cand_locs (list or tuple of lists): Candidate locations for the sensors. If tuple, position cand_locs should be
        first.
        beam (Beam): Beam object.
        dict (dict): Dictionary to save the results.
        p1p2pairs (list of tuples, optional): Pairs of (p1, p2) for combined sensor placement (position and strain).
    Returns:
        dict: Dictionary with optimal sensor placement results.
    """
    dict = {}
    dict[sensor_type] = {}

    if sensor_type == "Both":
        dict["Both"]["Position Candidate Locations"] = CommentedSeq(cand_locs[0])
        dict["Both"]["Position Candidate Locations"].fa.set_flow_style()
        dict["Both"]["Strain Candidate Locations"] = CommentedSeq(cand_locs[1])
        dict["Both"]["Strain Candidate Locations"].fa.set_flow_style()

        C_cand_pos = beam.range_measurement_matrix(cand_locs[0])
        C_cand_strain = beam.strain_measurement_matrix(cand_locs[1])
        C_cand = np.vstack((C_cand_pos, C_cand_strain))

        for pair in p1p2pairs:
            p1, p2 = pair
            dict["Both"][f"p1 = {p1}, p2 = {p2}"] = {}

            # Add baseline placement (equally spaced)
            dict["Both"][f"p1 = {p1}, p2 = {p2}"]["Baseline Indices Position"] = CommentedSeq(
                baseline_indices("Position", cand_locs[0], p1)
            )
            dict["Both"][f"p1 = {p1}, p2 = {p2}"]["Baseline Indices Position"].fa.set_flow_style()
            dict["Both"][f"p1 = {p1}, p2 = {p2}"]["Baseline Indices Strain"] = CommentedSeq(
                baseline_indices("Strain", cand_locs[1], p2)
            )
            dict["Both"][f"p1 = {p1}, p2 = {p2}"]["Baseline Indices Strain"].fa.set_flow_style()

            for obj_func in ["lambda_min", "trace", "log_det"]:
                selector_vector_1, selector_vector_2, objective_value = optimize_placement(
                    beam.A, C_cand_pos, p1, obj_func, C_cand_strain, p2
                )

                dict["Both"][f"p1 = {p1}, p2 = {p2}"][f"Maximizing {obj_func}"] = {}
                for selector_vector, sensor in zip((selector_vector_1, selector_vector_2), ("Position", "Strain")):
                    if selector_vector is None:
                        dict["Both"][f"p1 = {p1}, p2 = {p2}"][f"Maximizing {obj_func}"] = {
                            "Error": "Optimization failed."
                        }
                    else:
                        # check if selector_vector is binary
                        if is_binary(selector_vector):
                            opt_indices = np.nonzero(np.round(selector_vector))[0]
                            dict["Both"][f"p1 = {p1}, p2 = {p2}"][f"Maximizing {obj_func}"].update(
                                {f"Optimal Indices {sensor}": CommentedSeq(opt_indices.tolist())}
                            )
                            dict["Both"][f"p1 = {p1}, p2 = {p2}"][f"Maximizing {obj_func}"][
                                f"Optimal Indices {sensor}"
                            ].fa.set_flow_style()
                            if sensor == "Strain":
                                dict["Both"][f"p1 = {p1}, p2 = {p2}"][f"Maximizing {obj_func}"][
                                    "Optimized objective value"
                                ] = float(objective_value)
                        else:
                            formatted_vector = [float(f"{x:.6e}") for x in selector_vector[:, 0]]
                            dict["Both"][f"p1 = {p1}, p2 = {p2}"][f"Maximizing {obj_func}"].update(
                                {f"Non-binary selector vector {sensor}": CommentedSeq(formatted_vector)}
                            )
                            dict["Both"][f"p1 = {p1}, p2 = {p2}"][f"Maximizing {obj_func}"][
                                f"Non-binary selector vector {sensor}"
                            ].fa.set_flow_style()

    else:
        dict[sensor_type]["Candidate Locations"] = CommentedSeq(cand_locs)
        dict[sensor_type]["Candidate Locations"].fa.set_flow_style()
        if sensor_type == "Position":
            C_cand = beam.range_measurement_matrix(cand_locs)
        elif sensor_type == "Strain":
            C_cand = beam.strain_measurement_matrix(cand_locs)

        # Add baseline placement (equally spaced)
        for p in range(1, 4):
            dict[sensor_type][f"p = {p}"] = {}
        if sensor_type == "Position":
            dict["Position"]["p = 1"]["Baseline Indices"] = CommentedSeq([len(cand_locs) // 2])  # Last index
            dict["Position"]["p = 1"]["Baseline Indices"].fa.set_flow_style()
            dict["Position"]["p = 2"]["Baseline Indices"] = CommentedSeq([len(cand_locs) // 2, len(cand_locs) - 1])
            dict["Position"]["p = 2"]["Baseline Indices"].fa.set_flow_style()
            dict["Position"]["p = 3"]["Baseline Indices"] = CommentedSeq(
                [len(cand_locs) // 3, 2 * len(cand_locs) // 3, len(cand_locs) - 1]
            )
            dict["Position"]["p = 3"]["Baseline Indices"].fa.set_flow_style()
        elif sensor_type == "Strain":
            dict["Strain"]["p = 1"]["Baseline Indices"] = CommentedSeq([len(cand_locs) // 2])  # First index
            dict["Strain"]["p = 1"]["Baseline Indices"].fa.set_flow_style()
            dict["Strain"]["p = 2"]["Baseline Indices"] = CommentedSeq([0, len(cand_locs) // 2])
            dict["Strain"]["p = 2"]["Baseline Indices"].fa.set_flow_style()
            dict["Strain"]["p = 3"]["Baseline Indices"] = CommentedSeq(
                [0, len(cand_locs) // 3, 2 * len(cand_locs) // 3]
            )
            dict["Strain"]["p = 3"]["Baseline Indices"].fa.set_flow_style()

        for p in range(1, 4):
            for obj_func in ["lambda_min", "trace", "log_det"]:
                selector_vector, objective_value = optimize_placement(beam.A, C_cand, p, obj_func)

                # If the optimization failed, continue to the next iteration
                if selector_vector is None:
                    dict[sensor_type][f"p = {p}"][f"Maximizing {obj_func}"] = {"Error": "Optimization failed."}
                else:
                    # check if selector_vector is binary
                    if is_binary(selector_vector):
                        opt_indices = np.nonzero(np.round(selector_vector))[0]
                        dict[sensor_type][f"p = {p}"][f"Maximizing {obj_func}"] = {
                            "Optimal Indices": CommentedSeq(opt_indices.tolist()),
                            "Optimized objective value": float(objective_value),
                        }
                        dict[sensor_type][f"p = {p}"][f"Maximizing {obj_func}"]["Optimal Indices"].fa.set_flow_style()
                    else:
                        formatted_vector = [float(f"{x:.6e}") for x in selector_vector[:, 0]]
                        dict[sensor_type][f"p = {p}"][f"Maximizing {obj_func}"] = {
                            "Non-binary selector vector": CommentedSeq(formatted_vector),
                        }
                        dict[sensor_type][f"p = {p}"][f"Maximizing {obj_func}"][
                            "Non-binary selector vector"
                        ].fa.set_flow_style()

                # add objective value for baseline placement
                bas_value = compute_gramian_metric(
                    beam.A,
                    C_cand,
                    dict[sensor_type][f"p = {p}"]["Baseline Indices"],
                    obj_func,
                )
                dict[sensor_type][f"p = {p}"][f"Maximizing {obj_func}"]["Baseline placement objective value"] = float(
                    bas_value
                )
    return dict


placement_dict = {}

cand_locs_pos = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]  # [m]
cand_locs_str = np.arange(0, 0.13, 0.01).tolist()  # [m]

# Position sensors
placement_dict.update(process_optimal_placement("Position", cand_locs_pos, beam))

# Strain sensors
placement_dict.update(process_optimal_placement("Strain", cand_locs_str, beam))

# Combined sensors
placement_dict.update(process_optimal_placement("Both", (cand_locs_pos[:-2], cand_locs_str[:-4]), beam, [(1, 3)]))
# Adjust baseline indices for combined sensors since there is a constraint on the placement
# Can still place position sensor in the middle of the segment, doesn't need to be in the middle of the available locs
placement_dict["Both"]["p1 = 1, p2 = 3"]["Baseline Indices Position"] = CommentedSeq([3])
placement_dict["Both"]["p1 = 1, p2 = 3"]["Baseline Indices Position"].fa.set_flow_style()
placement_dict["Both"]["p1 = 1, p2 = 3"]["Baseline Indices Strain"] = CommentedSeq([0, 4, 8])
placement_dict["Both"]["p1 = 1, p2 = 3"]["Baseline Indices Strain"].fa.set_flow_style()

# Save results to YAML file
with open(f"results/sensor_placement_results.yaml", "w") as file:
    yaml.dump(placement_dict, file)

# Plot the Gramian metrics for each sensor type
for sensor_type in placement_dict:
    if sensor_type != "Both":
        gram_dicts = []
        for p_case in (item for item in placement_dict[sensor_type] if "p =" in item):
            gram_values = {}
            for obj_func in (item for item in placement_dict[sensor_type][p_case] if "Maximizing" in item):
                if any(k.startswith("Optimal Indices") for k in placement_dict[sensor_type][p_case][obj_func]):
                    gram_values[(p_case, obj_func)] = (
                        placement_dict[sensor_type][p_case][obj_func]["Optimized objective value"],
                        placement_dict[sensor_type][p_case][obj_func]["Baseline placement objective value"],
                    )
            gram_dicts.append(gram_values)
        # Plot the Gramian metrics
        num_axes = max(len(d) for d in gram_dicts)
        fig, axes = plt.subplots(nrows=1, ncols=num_axes, figsize=(4 * num_axes, 4))

        # group by objective function
        obj_funcs = set(item[1] for d in gram_dicts for item in d.keys() if "Maximizing" in item[1])
        order = ["lambda_min", "trace", "log_det"]
        obj_funcs = sorted(
            obj_funcs,
            key=lambda x: order.index(x.replace("Maximizing ", ""))
            if x.replace("Maximizing ", "") in order
            else len(order),
        )
        for i, obj_func in enumerate(obj_funcs):
            for d in gram_dicts:
                for item in d.keys():
                    if item[1] == obj_func:
                        p = int(item[0].replace("p = ", ""))
                        axes[i].scatter(p, d[item][0], zorder=2, color="tab:blue")
                        if np.isclose(d[item][0], d[item][1], rtol=1e-3):
                            axes[i].scatter(p, d[item][1], marker="D", zorder=1, s=120, color="tab:orange")
                        else:
                            axes[i].scatter(p, d[item][1], marker="D", zorder=1, color="tab:orange")
            if "lambda_min" in obj_func:
                axes[i].set_ylabel(r"$\lambda_{\min}(\mathbf{W}_{\mathrm{o}})$")
            elif "trace" in obj_func:
                axes[i].set_ylabel(r"$\mathrm{tr}(\mathbf{W}_{\mathrm{o}})$")
            elif "log_det" in obj_func:
                axes[i].set_ylabel(r"$\log(\det(\mathbf{W}_{\mathrm{o}}))$")
            axes[i].grid(True)

            # add legend
            if i == len(obj_funcs) - 1:
                axes[i].scatter([], [], zorder=2, color="tab:blue", label="Optimal")
                axes[i].scatter([], [], marker="D", zorder=1, color="tab:orange", label="Baseline")
                axes[i].legend(loc="lower right", borderpad=0.1, handlelength=0.7, borderaxespad=0.2)
        if num_axes == 2:
            fig.supxlabel("Number of sensors $p$", fontsize=28)
        elif num_axes == 3:
            axes[1].set_xlabel("Number of sensors $p$")
        fig.savefig(f"results/gramian_metrics_{sensor_type.lower()}.pdf", bbox_inches="tight")
