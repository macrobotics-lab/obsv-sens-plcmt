"""
plot_rmse.py
"""

import pickle
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from ruamel.yaml import YAML

# Use style sheet for plotting
plt.style.use("my_default.mplstyle")

# load target directories
yaml = YAML()


def plot_rmse_vs_time(results_pkl):
    """
    Plot RMSE vs time from the results pickle file.

    Args:
        results_pkl (str): Path to the results pickle file.

    Returns:
        None
    """
    with open(results_pkl, "rb") as file:
        results_dict = pickle.load(file)

    for key, value in results_dict.items():
        # filter out ground truth keys and baseline keys and both sensor types trials
        if "Ground Truth" not in key and "Baseline" not in key and "Both" not in key:
            # check if it is a sine input or step input
            if "A" in key[0]:
                input_type = "sine"
            else:
                input_type = "step"

            # get sensor type
            sensor_type = key[1]

            # get number of active sensors
            num_sensors = key[2].replace(" ", "").replace("=", "")

            # get objective function
            obj_func = key[-1]

            # create directory if it doesn't exist
            plot_dir = f"results/kf_results/plots/{sensor_type}/{input_type}/{num_sensors}/{obj_func}"
            Path(plot_dir).mkdir(parents=True, exist_ok=True)

            # plot RMSE vs time (optimized vs baseline)
            rmse_opt = value["rmse"]
            key_bas = key[:-1] + ("Baseline",)
            rmse_bas = results_dict[key_bas]["rmse"]

            fig, ax = plt.subplots()
            ax.plot(np.linspace(0, 10, len(rmse_opt)), rmse_opt, label="Optimized")
            ax.plot(np.linspace(0, 10, len(rmse_bas)), rmse_bas, label="Baseline", linestyle="--")
            ax.legend()
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("RMSE [mm]")
            ax.grid(True)
            fig.savefig(f"{plot_dir}/{key[0]}.pdf", bbox_inches="tight")
            plt.close()


def plot_representative_rmse_vs_time(top_plot_pkl, top_plot_key, bottom_plot_pkl, bottom_plot_key):
    """
    Create a representative plot with RMSE vs time with two subplots.
    """
    with open(top_plot_pkl, "rb") as file:
        top_results = pickle.load(file)
    top_rmse_opt = top_results[top_plot_key]["rmse"]
    top_rmse_bas = top_results[top_plot_key[:-1] + ("Baseline",)]["rmse"]

    with open(bottom_plot_pkl, "rb") as file:
        bottom_results = pickle.load(file)
    bottom_rmse_opt = bottom_results[bottom_plot_key]["rmse"]
    bottom_rmse_bas = bottom_results[bottom_plot_key[:-1] + ("Baseline",)]["rmse"]

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top plot
    axs[0].plot(np.linspace(0, 10, top_rmse_opt.shape[0]), top_rmse_opt, label="Optimal")
    axs[0].plot(np.linspace(0, 10, top_rmse_bas.shape[0]), top_rmse_bas, label="Baseline", linestyle="--")
    axs[0].legend(loc="best", borderpad=0.1, borderaxespad=0.15)
    axs[0].set_ylabel("RMSE [mm]")
    axs[0].grid(True)

    # Bottom plot
    axs[1].plot(np.linspace(0, 10, bottom_rmse_opt.shape[0]), bottom_rmse_opt, label="Optimal")
    axs[1].plot(np.linspace(0, 10, bottom_rmse_bas.shape[0]), bottom_rmse_bas, label="Baseline", linestyle="--")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("RMSE [mm]")
    axs[1].grid(True)

    fig_name = f"rep_trials_{top_plot_key[0]}_{top_plot_key[1].lower()[:3]}_{top_plot_key[2].replace(' = ', '')}_{top_plot_key[3]}_{bottom_plot_key[0]}_{bottom_plot_key[1].lower()[:3]}_{bottom_plot_key[2].replace(' = ', '')}_{bottom_plot_key[3]}"  # noqa: E501
    fig.savefig(f"results/kf_results/plots/{fig_name}.pdf")


def extract_rmse_traj_dict(results_dict):
    """
    Extracts and organizes total trajectory RMSEs from a results dictionary.
    Args:
        results_dict (dict): Dictionary containing results, where each key is a tuple and each value is a dictionary
        with an "rmse_traj" entry.
    Returns:
        dict or tuple of dicts: A dictionary with keys as tuples of (sensor_type, num_sensors, obj_func) and rmse values
        as lists. Returns a tuple of dictionaries if the input is a sine input, containing amplitude and frequency
        specific RMSEs.
    """
    rmse_traj_dict = {}
    rmse_traj_dict_amp = {}
    rmse_traj_dict_freq = {}
    for key, value in results_dict.items():
        if "Ground Truth" not in key:
            sensor_type = key[1]
            num_sensors = key[2]
            obj_func = key[-1]
            # replace obj_func with proper labels
            if obj_func == "lambda_min":
                obj_func = r"$\lambda_{\min}(\mathbf{W}_{\mathrm{o}})$"
            elif obj_func == "trace":
                obj_func = r"$\mathrm{tr}(\mathbf{W}_{\mathrm{o}})$"
            elif obj_func == "log_det":
                obj_func = r"$\log(\det(\mathbf{W}_{\mathrm{o}}))$"

            rmse_traj_dict.setdefault((sensor_type, num_sensors, obj_func), []).append(value["rmse_traj"])

            if "A" in key[0]:
                amp = key[0][1:].split("kPa_w")[0]
                freq = key[0][1:].split("kPa_w")[1]
                rmse_traj_dict_amp.setdefault((sensor_type, num_sensors, obj_func, amp), []).append(value["rmse_traj"])
                rmse_traj_dict_freq.setdefault((sensor_type, num_sensors, obj_func, freq), []).append(
                    value["rmse_traj"]
                )  # noqa: E501

    if "A" in key[0]:
        # If the input is a sine input, return the amplitude and frequency dictionaries
        return rmse_traj_dict, rmse_traj_dict_amp, rmse_traj_dict_freq
    return rmse_traj_dict


def split_by_sensor_type(rmse_traj_dict):
    pos_dicts = {}
    str_dicts = {}
    for p in ["p = 1", "p = 2", "p = 3"]:
        pos_dicts[p] = {key[2]: value for key, value in rmse_traj_dict.items() if key[0] == "Position" and key[1] == p}
        str_dicts[p] = {key[2]: value for key, value in rmse_traj_dict.items() if key[0] == "Strain" and key[1] == p}

    # Create empty dictionary for both sensor types configuration
    both_dicts = {}
    p1p2_pairs = set(key[1] for key in rmse_traj_dict.keys() if key[0] == "Both")
    for p1p2_pair in p1p2_pairs:
        both_dicts[p1p2_pair] = {
            key[2]: value for key, value in rmse_traj_dict.items() if key[0] == "Both" and key[1] == p1p2_pair
        }
    return pos_dicts, str_dicts, both_dicts


def plot_single_boxplot(ax, data_dict, ylabel=None, title=None):
    """
    Helper function to create a single boxplot on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        data_dict (dict): Dictionary containing data for the boxplot.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.

    Returns:
        None
    """
    boxplot = ax.boxplot(
        data_dict.values(),
        patch_artist=True,
        widths=0.5,
    )
    # Set colors
    for i, box in enumerate(boxplot["boxes"]):
        if i == len(boxplot["boxes"]) - 1:
            box.set_facecolor("peachpuff")
        else:
            box.set_facecolor("lightblue")

    for i, median in enumerate(boxplot["medians"]):
        if i == len(boxplot["medians"]) - 1:
            median.set_color("tab:orange")
        else:
            median.set_color("tab:blue")
    ax.set_xticklabels(data_dict.keys(), rotation=45, ha="right")
    ax.grid(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def compute_stats(dict):
    """
    Computes statistics for each key in the input dictionary compared to the "Baseline" key.
    test.
    Parameters:
        dict (dict): A dictionary where each key maps to a list of numerical values (RMSEs).
                     The dictionary must contain a "Baseline" key for comparison.
    Returns:
        dict: A dictionary mapping each key (except "Baseline") to its corresponding statistics (dict)
              from the Wilcoxon test. If the p-value test cannot be computed, the value is None.
    """

    stats = {}
    for key, value in dict.items():
        stats[key] = {}
        try:
            p_value = float(sp.stats.wilcoxon(value, dict["Baseline"])[1])
        except ValueError as e:
            p_value = None
        stats[key]["p-value"] = p_value
        stats[key]["median"] = round(float(np.median(value)), 2)
        stats[key]["Q1"] = round(float(np.percentile(value, 25)), 2)
        stats[key]["Q3"] = round(float(np.percentile(value, 75)), 2)
    return stats


def plot_rmse_box_plots_helper(step_dicts, sine_dicts, fig_name):
    """
    Helper function to plot RMSE box plots for step and sine results.

    Args:
        step_dicts (dict): Dictionary containing step RMSE data.
        sine_dicts (dict): Dictionary containing sine RMSE data.
        figname (str): Name of the figure file to save.

    Returns:
        None
    """
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(3, 3, width_ratios=[0.3, 1, 1])

    stats_dict = {"step": {}, "sine": {}}
    # Step input plots
    for i, p in enumerate(["p = 1", "p = 2", "p = 3"]):
        ax = fig.add_subplot(gs[i, 1])
        plot_single_boxplot(ax, step_dicts[p], ylabel="RMSE [mm]", title="Step input" if i == 0 else None)
        stats_dict["step"][f"{p}"] = compute_stats(step_dicts[p])

    # Sine input plots
    for i, p in enumerate(["p = 1", "p = 2", "p = 3"]):
        ax = fig.add_subplot(gs[i, 2])
        plot_single_boxplot(ax, sine_dicts[p], title="Sinusoidal input" if i == 0 else None)
        stats_dict["sine"][f"{p}"] = compute_stats(sine_dicts[p])

    # Row labels
    for i, p in enumerate(["p=1", "p=2", "p=3"]):
        ax_label = fig.add_subplot(gs[i, 0])
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, f"${p}$", va="center", ha="center", fontsize=28, rotation="horizontal")

    fig.savefig(f"results/kf_results/plots/{fig_name}.pdf", bbox_inches="tight")
    # save statistics for each key
    with open(f"results/kf_results/plots/{fig_name}_stats.yaml", "w") as file:
        yaml.dump(stats_dict, file)


def plot_single_boxplot_amp_freq(ax, data_dict, baseline_data_dict, x_axis, banner_fig=False):
    boxplot = ax.boxplot(
        [item for pair in zip(data_dict.values(), baseline_data_dict.values()) for item in pair],
        patch_artist=True,
        widths=0.5,
    )
    # Set colors
    for i, box in enumerate(boxplot["boxes"]):
        box.set_facecolor("lightblue" if i % 2 == 0 else "peachpuff")
    for i, median in enumerate(boxplot["medians"]):
        median.set_color("tab:blue" if i % 2 == 0 else "tab:orange")
    if x_axis == "amplitude":
        ax.set_ylabel("RMSE [mm]")

    # xlabels
    if x_axis == "amplitude":
        label_prefix = "A"
    elif x_axis == "frequency":
        label_prefix = r"\omega"
    ax.set_xticks(np.arange(1.5, 2 * len(data_dict), 2))
    if banner_fig:
        ax.set_xticklabels([rf"${key}$" for key in data_dict.keys()])
    else:
        ax.set_xticklabels([rf"${label_prefix} = {key}$" for key in data_dict.keys()], rotation=45)

    # minor ticks for grid
    ax.set_xticks(np.arange(2.5, len(data_dict) * 2, 2), minor=True)
    ax.grid(True, axis="x", which="minor")
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", which="minor", length=0)


def plot_helper_amp_freq(amp_data, amp_baseline_data, freq_data, freq_baseline_data, legend_label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_single_boxplot_amp_freq(ax1, amp_data, amp_baseline_data, "amplitude")
    legend_elements = [
        matplotlib.patches.Patch(facecolor="lightblue", label=legend_label),
        matplotlib.patches.Patch(facecolor="peachpuff", label="Baseline"),
    ]
    ax1.legend(handles=legend_elements, loc="best")

    plot_single_boxplot_amp_freq(ax2, freq_data, freq_baseline_data, "frequency")
    return fig


def plot_rmse_box_plots_helper_amp_freq(rmse_dict_amp, rmse_dict_freq):
    # lists to hold data for amplitude and frequency plots (banner figure)
    amp_data_dicts = []
    freq_data_dicts = []
    for sensor_type in ["Position", "Strain"]:
        for p in ["p = 1", "p = 2", "p = 3"]:
            for obj_func in ["lambda", "tr", "log"]:
                amp_data = {}
                amp_baseline_data = {}
                for key, value in rmse_dict_amp.items():
                    if key[0] == sensor_type and key[1] == p:
                        if obj_func in key[2]:
                            amp_data[key[3]] = value
                            legend_label = key[2]
                        elif "Baseline" in key[2]:
                            amp_baseline_data[key[3]] = value

                freq_data = {}
                freq_baseline_data = {}
                for key, value in rmse_dict_freq.items():
                    if key[0] == sensor_type and key[1] == p:
                        if obj_func in key[2]:
                            freq_data[key[3]] = value
                        elif "Baseline" in key[2]:
                            freq_baseline_data[key[3]] = value

                if amp_data:
                    fig = plot_helper_amp_freq(amp_data, amp_baseline_data, freq_data, freq_baseline_data, legend_label)
                    amp_p_values = compute_stats_amp_freq(amp_data, amp_baseline_data)
                    freq_p_values = compute_stats_amp_freq(freq_data, freq_baseline_data)

                    if sensor_type == "Position" and p == "p = 3" and obj_func == "log":
                        amp_data_dicts.append(amp_data)
                        amp_data_dicts.append(amp_baseline_data)
                        freq_data_dicts.append(freq_data)
                        freq_data_dicts.append(freq_baseline_data)
                    if sensor_type == "Strain" and p == "p = 3" and obj_func == "tr":
                        amp_data_dicts.append(amp_data)
                        amp_data_dicts.append(amp_baseline_data)
                        freq_data_dicts.append(freq_data)
                        freq_data_dicts.append(freq_baseline_data)

                    if obj_func == "lambda":
                        folder = "lambda_min"
                    elif obj_func == "tr":
                        folder = "trace"
                    elif obj_func == "log":
                        folder = "log_det"
                    # create directory if it doesn't exist
                    plot_dir = f"results/kf_results/plots/{sensor_type}/sine/{p.replace(' = ', '')}/{folder}"
                    Path(plot_dir).mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        f"{plot_dir}/rmse_traj_amp_freq.pdf",  # noqa: E501
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                    stats_dict = {"amp": amp_p_values, "freq": freq_p_values}
                    # save p-values for statistical tests
                    with open(f"{plot_dir}/p_values.yaml", "w") as file:
                        yaml.dump(stats_dict, file)

    for p1p2_pair in set(key[1] for key in rmse_dict_amp.keys() if key[0] == "Both"):
        for obj_func in ["lambda", "tr", "log"]:
            amp_data = {}
            amp_baseline_data = {}
            for key, value in rmse_dict_amp.items():
                if key[1] == p1p2_pair:
                    if obj_func in key[2]:
                        amp_data[key[3]] = value
                        legend_label = key[2]
                    elif "Baseline" in key[2]:
                        amp_baseline_data[key[3]] = value

            freq_data = {}
            freq_baseline_data = {}
            for key, value in rmse_dict_freq.items():
                if key[1] == p1p2_pair:
                    if obj_func in key[2]:
                        freq_data[key[3]] = value
                    elif "Baseline" in key[2]:
                        freq_baseline_data[key[3]] = value

            if amp_data:
                fig = plot_helper_amp_freq(amp_data, amp_baseline_data, freq_data, freq_baseline_data, legend_label)
                amp_p_values = compute_stats_amp_freq(amp_data, amp_baseline_data)
                freq_p_values = compute_stats_amp_freq(freq_data, freq_baseline_data)

                if p1p2_pair == "p1 = 1, p2 = 3" and obj_func == "tr":
                    amp_data_dicts.append(amp_data)
                    amp_data_dicts.append(amp_baseline_data)
                    freq_data_dicts.append(freq_data)
                    freq_data_dicts.append(freq_baseline_data)

                if obj_func == "lambda":
                    folder = "lambda_min"
                elif obj_func == "tr":
                    folder = "trace"
                elif obj_func == "log":
                    folder = "log_det"
                # create directory if it doesn't exist
                parts = p1p2_pair.replace(" ", "").split(",")
                p1 = int(parts[0].split("=")[1])
                p2 = int(parts[1].split("=")[1])
                plot_dir = f"results/kf_results/plots/Both/pos{p1}_str{p2}/sine/{folder}"
                Path(plot_dir).mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{plot_dir}/rmse_traj_amp_freq.pdf", bbox_inches="tight")
                plt.close(fig)
                stats_dict = {"amp": amp_p_values, "freq": freq_p_values}
                # save p-values for statistical tests
                with open(f"{plot_dir}/p_values.yaml", "w") as file:
                    yaml.dump(stats_dict, file)
    return amp_data_dicts, freq_data_dicts


def compute_stats_amp_freq(data_dict, baseline_data_dict):
    """
    Compute p-values for amplitude and frequency RMSE data using the Wilcoxon signed-rank test.

    Args:
        data_dict (dict): Dictionary containing amplitude or frequency RMSE data.
        baseline_data_dict (dict): Dictionary containing baseline RMSE data.

    Returns:
        dict: A dictionary with keys as the amplitude or frequency values and p-values as values.
    """

    p_values = {}
    for key in data_dict.keys():
        try:
            p_value = float(sp.stats.wilcoxon(data_dict[key], baseline_data_dict[key])[1])
        except ValueError as e:
            p_value = None
        p_values[key] = p_value
    return p_values


def plot_amp_freq_box_plots_all(amp_data_dicts, freq_data_dicts):
    """
    Create banner figure with amplitude and frequency box plots for selected configurations.
    """
    fig = plt.figure(figsize=(26, 12))
    gs = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.3)

    for i, (title, obj_func) in enumerate(
        [
            ("Position", r"$\log(\det(\mathbf{W}_{\mathrm{o}}))$"),
            ("Strain", r"$\mathrm{tr}(\mathbf{W}_{\mathrm{o}})$"),
            ("Combined", r"$\mathrm{tr}(\mathbf{W}_{\mathrm{o}})$"),
        ]
    ):
        ax1 = fig.add_subplot(gs[0, i])
        ax2 = fig.add_subplot(gs[1, i])

        plot_single_boxplot_amp_freq(
            ax1, amp_data_dicts[2 * i], amp_data_dicts[2 * i + 1], "amplitude", banner_fig=True
        )
        plot_single_boxplot_amp_freq(
            ax2, freq_data_dicts[2 * i], freq_data_dicts[2 * i + 1], "frequency", banner_fig=True
        )

        ax1.set_title(f"{title}")

        if i == 0:
            ax1.set_ylabel("RMSE [mm]")
            ax2.set_ylabel("RMSE [mm]")
        else:
            ax1.set_ylabel("")
            ax2.set_ylabel("")
        if i == 1:
            ax1.set_xlabel(r"Amplitude $A$ [kPa]")
            ax2.set_xlabel(r"Frequency $\omega$ [rad/s]")

        legend_elements = [
            matplotlib.patches.Patch(facecolor="lightblue", label=obj_func),
            matplotlib.patches.Patch(facecolor="peachpuff", label="Baseline"),
        ]
        ax1.legend(handles=legend_elements, loc="best")
    fig.savefig("results/kf_results/plots/rmse_traj_amp_freq_banner.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rmse_box_plots(step_results_pkl, sine_results_pkl):
    """
    Generate box plots comparing RMSE results for step and sine experiments.
    Args:
        step_results_pkl (str): File path to the pickle file containing step experiment results.
        sine_results_pkl (str): File path to the pickle file containing sine experiment results.
    Returns:
        None: The function generates and displays box plots but does not return any value.
    """
    with open(step_results_pkl, "rb") as file:
        step_results_dict = pickle.load(file)
    step_rmse_traj_dict = extract_rmse_traj_dict(step_results_dict)
    step_pos_dicts, step_str_dicts, step_both_dicts = split_by_sensor_type(step_rmse_traj_dict)

    with open(sine_results_pkl, "rb") as file:
        sine_results_dict = pickle.load(file)
    sine_rmse_traj_dict, rmse_traj_dict_amp, rmse_traj_dict_freq = extract_rmse_traj_dict(sine_results_dict)
    sine_pos_dicts, sine_str_dicts, sine_both_dicts = split_by_sensor_type(sine_rmse_traj_dict)

    plot_rmse_box_plots_helper(step_pos_dicts, sine_pos_dicts, "Position/rmse_traj_pos")
    plot_rmse_box_plots_helper(step_str_dicts, sine_str_dicts, "Strain/rmse_traj_str")
    amp_data_dicts, freq_data_dicts = plot_rmse_box_plots_helper_amp_freq(rmse_traj_dict_amp, rmse_traj_dict_freq)
    plot_amp_freq_box_plots_all(amp_data_dicts, freq_data_dicts)

    # plot box plots for combined sensor types
    for p1p2_pair in step_both_dicts.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_single_boxplot(ax1, step_both_dicts[p1p2_pair], ylabel="RMSE [mm]", title="Step input")
        plot_single_boxplot(ax2, sine_both_dicts[p1p2_pair], title="Sinusoidal input")

        stats_dict = {p1p2_pair: {}}
        stats_dict[p1p2_pair]["step"] = compute_stats(step_both_dicts[p1p2_pair])
        stats_dict[p1p2_pair]["sine"] = compute_stats(sine_both_dicts[p1p2_pair])

        # create directory if it doesn't exist
        parts = p1p2_pair.replace(" ", "").split(",")
        p1 = int(parts[0].split("=")[1])
        p2 = int(parts[1].split("=")[1])
        plot_dir = f"results/kf_results/plots/Both/pos{p1}_str{p2}"
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"results/kf_results/plots/Both/pos{p1}_str{p2}/rmse_traj_both.pdf", bbox_inches="tight")
        plt.close(fig)
        # save statistics for each key
        with open(f"results/kf_results/plots/Both/pos{p1}_str{p2}/rmse_traj_both_stats.yaml", "w") as file:
            yaml.dump(stats_dict, file)


plot_rmse_vs_time("results/kf_results/sine_kf_results.pkl")
plot_rmse_vs_time("results/kf_results/step_kf_results.pkl")
plot_rmse_box_plots("results/kf_results/step_kf_results.pkl", "results/kf_results/sine_kf_results.pkl")

plot_representative_rmse_vs_time(
    "results/kf_results/step_kf_results.pkl",
    ("90kPa", "Position", "p = 1", "trace"),
    "results/kf_results/sine_kf_results.pkl",
    ("A40kPa_w3", "Position", "p = 3", "log_det"),
)
