"""
sensitivity_analysis.py
"""

import os

import numpy as np
from ruamel.yaml import YAML

yaml = YAML()
with open("results/params.yaml", "r") as file:
    params = yaml.load(file)
with open("results/sensor_placement_results.yaml", "r") as file:
    spr = yaml.load(file)


def perturb_param(param_name, perturbation):
    with open("results/params.yaml", "r") as file:
        new_params = yaml.load(file)
    if param_name == "zetas":
        new_params["beam"]["zetas"] = [z * (1 + perturbation) for z in params["beam"]["zetas"]]
    else:
        new_params["beam"][param_name] = params["beam"][param_name] * (1 + perturbation)
    with open("results/params.yaml", "w") as file:
        yaml.dump(new_params, file)


def check_for_changes(spr, new_spr):
    for sensor_type in ["Position", "Strain"]:
        for p in [1, 2, 3]:
            for obj_func, result in spr[sensor_type][f"p = {p}"].items():
                if "Optimal Indices" in result:
                    if "Optimal Indices" not in new_spr[sensor_type][f"p = {p}"][obj_func]:
                        print(f"Change detected in {sensor_type}, p = {p}, {obj_func}")
                        print("Old:", result)
                        print("New:", new_spr[sensor_type][f"p = {p}"][obj_func])
                        return True
                    elif result["Optimal Indices"] != new_spr[sensor_type][f"p = {p}"][obj_func]["Optimal Indices"]:
                        print(f"Change detected in {sensor_type}, p = {p}, {obj_func}")
                        print("Old:", result)
                        print("New:", new_spr[sensor_type][f"p = {p}"][obj_func])
                        return True
                elif "Non-binary selector vector" in result:
                    if "Non-binary selector vector" not in new_spr[sensor_type][f"p = {p}"][obj_func]:
                        print(f"Change detected in {sensor_type}, p = {p}, {obj_func}")
                        print("Old:", result)
                        print("New:", new_spr[sensor_type][f"p = {p}"][obj_func])
                        return True
    for p1p2_pair in [key for key in spr["Both"].keys() if "p1" in key]:
        for obj_func, result in spr["Both"][p1p2_pair].items():
            for sensor_type in ["Position", "Strain"]:
                if f"Optimal Indices {sensor_type}" in result:
                    if (
                        result[f"Optimal Indices {sensor_type}"]
                        != new_spr["Both"][p1p2_pair][obj_func][f"Optimal Indices {sensor_type}"]
                    ):
                        print(f"Change detected in Both, {p1p2_pair}, {obj_func}, {sensor_type}")
                        print("Old:", result)
                        print("New:", new_spr["Both"][p1p2_pair][obj_func])
                        return True
                elif f"Non-binary selector vector {sensor_type}" in result:
                    if f"Non-binary selector vector {sensor_type}" not in new_spr["Both"][p1p2_pair][obj_func]:
                        print(f"Change detected in Both, {p1p2_pair}, {obj_func}, {sensor_type}")
                        print("Old:", result)
                        print("New:", new_spr["Both"][p1p2_pair][obj_func])
                        return True
    return False

for param in ["EI", "rho", "zetas"]:
    for sign in [1, -1]:
        for perturbation in np.linspace(0, 1, 101):
            perturb_param(param, perturbation * sign)
            os.system("python optimization.py")
            with open("results/sensor_placement_results.yaml", "r") as file:
                new_spr = yaml.load(file)
            
            # Reset the parameters back to the original values
            with open("results/params.yaml", "w") as file:
                yaml.dump(params, file)
            # Reset the sensor placement results back to the original values
            with open("results/sensor_placement_results.yaml", "w") as file:
                yaml.dump(spr, file)

            if check_for_changes(spr, new_spr):
                print(f"Changes detected for {param} with perturbation {perturbation * sign}")
                break
