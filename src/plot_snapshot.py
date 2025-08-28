"""
plot_snapshot.py

Usage:
    Run this script from the directory where it is located.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

# Use style sheet for plotting
plt.style.use("my_default.mplstyle")


def plot_snapshot_measured(time, input_type, p_set, w=None):
    """
    Plot the measured 2D position of all 7 centroids.

    Args:
        time (float): Time of the measurement [s].
        input_type (str): "impulse", "sine", or "step".
        p_set (int): Set pressure during the step response [kPa], or in the case of sine
        inputs, the amplitude [kPa].
        w (int): Only used for sine inputs. The input frequency [rad/s] (1-6).

    Returns:
        None
    """
    # load target directories
    yaml = YAML()
    with open("directories.yaml", "r") as file:
        directories = yaml.load(file)
    data_dir = directories["data_dir"]

    if input_type == "sine":
        df = pd.read_csv(f"{data_dir}/{input_type}/A{p_set}kPa_w{w}.csv")
    else:
        df = pd.read_csv(f"{data_dir}/{input_type}/{p_set}kPa.csv")

    spine2D = np.zeros((len(df["Time"]), 2, 7))
    for i in range(7):
        spine2D[:, 0, i] = df[f"Spine2D{i + 1}_x"]
        spine2D[:, 1, i] = df[f"Spine2D{i + 1}_y"]

    idx = df.index[np.isclose(df["Time"], time, atol=0.01)][0]

    fig, ax = plt.subplots()
    ax.plot(spine2D[idx, 0, :], spine2D[idx, 1, :], marker="o", color="tab:blue")
    ax.grid(True)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.show()


def plot_snapshot_results(
    sensor_type, p_active, obj_func, time, input_type, p_set, w=None
):
    """
    Plot the robot shape from ground truth, baseline, and optimized sensor placement at
    a specific time.

    Args:
        sensor_type (str): Type of sensor ("Position" or "Strain").
        p_active (int): Number of active sensors.
        obj_func (str): Objective function used for optimization ("lambda_min", "trace",
        "log_det").
        time (float): Time of the measurement [s] (between 0 and 10 seconds).
        input_type (str): "sine", or "step".
        p_set (int): Set pressure during the step response [kPa], or in the case of sine
        inputs, the amplitude [kPa].
        w (int): Only used for sine inputs. The input frequency [rad/s] (1-6).
    """

    # get time index
    if input_type == "sine":
        df = pd.read_csv(f"../data/{input_type}/A{p_set}kPa_w{w}.csv")
    else:
        df = pd.read_csv(f"../data/{input_type}/{p_set}kPa.csv")
    idx = df.index[np.isclose(df["Time"], time, atol=0.01)][0]

    # load results pickle file
    results_pkl = f"results/kf_results/{input_type}_kf_results.pkl"
    with open(results_pkl, "rb") as file:
        results_dict = pickle.load(file)

    # create key for the results dictionary
    if input_type == "sine":
        first_entry = f"A{p_set}kPa_w{w}"
    elif input_type == "step":
        first_entry = f"{p_set}kPa"
    key = (first_entry, sensor_type, f"p = {p_active}", obj_func)
    key_baseline = key[:-1] + ("Baseline",)

    opt_coords = results_dict[key]["est_coords"]  # optimized placement coordinates
    gt_coords = results_dict[key]["gt_coords"]  # Ground truth coordinates
    bas_coords = results_dict[key_baseline][
        "est_coords"
    ]  # Baseline placement coordinates

    # get coordinates of centroids of measured markers
    num_centroids = 7
    spine2D = np.zeros((2, num_centroids))
    for i in range(num_centroids):
        spine2D[0, i] = df[f"Spine2D{i + 1}_x"][idx]
        spine2D[1, i] = df[f"Spine2D{i + 1}_y"][idx]

    fig, ax = plt.subplots()
    # Plot optimized placement
    ax.plot(
        opt_coords[: opt_coords.shape[0] // 2, idx],
        opt_coords[opt_coords.shape[0] // 2 :, idx],
        marker="o",
        color="tab:blue",
        label="Optimized Placement",
    )
    # Plot baseline placement
    ax.plot(
        bas_coords[: bas_coords.shape[0] // 2, idx],
        bas_coords[bas_coords.shape[0] // 2 :, idx],
        marker="o",
        color="tab:orange",
        label="Baseline Placement",
    )
    # Plot ground truth
    ax.plot(
        gt_coords[: gt_coords.shape[0] // 2, idx],
        gt_coords[gt_coords.shape[0] // 2 :, idx],
        marker="o",
        color="tab:green",
        label="Ground Truth",
    )
    # Plot measured centroids (markers)
    ax.plot(
        spine2D[0, :],
        spine2D[1, :],
        marker="o",
        color="tab:red",
        label="Measured Centroids",
    )
    ax.grid(True)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend()

    figname = f"results/kf_results/plots/{sensor_type}/{input_type}/p{p_active}/{obj_func}/snapshot_{time}s_{p_set}kPa"  # noqa: E501
    if w is not None:
        figname += f"_w{w}"
    # fig.savefig(f"{figname}.pdf", bbox_inches="tight")
    plt.show()


# Example usage
# plot_snapshot_measured(5, "step", 30)
# plot_snapshot_results("Position", 1, "trace", 6, "step", 60)
# plot_snapshot_results("Position", 3, "trace", 0.1, "sine", 30, 1)
