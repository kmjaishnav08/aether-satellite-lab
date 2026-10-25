"""Experiment comparison helpers and tidy export functions."""
from __future__ import annotations

from dataclasses import replace
import pandas as pd

from physics_model import SimulationInputs, run_simulation


def compare_approach_speed(base: SimulationInputs) -> pd.DataFrame:
    factors = (0.5, 1.0, 1.5)
    labels = ("Lower approach speed", "Current approach speed", "Higher approach speed")
    rows = []
    for label, factor in zip(labels, factors):
        case = replace(base, approach_speed_mps=max(0.0, base.approach_speed_mps * factor))
        out = run_simulation(case)
        rows.append({"Case": label, "Approach speed (m/s)": case.approach_speed_mps, "Collision energy loss (J)": out.collision_energy_loss_j})
    return pd.DataFrame(rows)


def compare_stiffness(base: SimulationInputs) -> pd.DataFrame:
    factors = (0.5, 1.0, 1.5)
    labels = ("Lower stiffness", "Current stiffness", "Higher stiffness")
    rows = []
    for label, factor in zip(labels, factors):
        case = replace(base, stiffness_n_per_m=max(1.0, base.stiffness_n_per_m * factor))
        out = run_simulation(case)
        rows.append({"Case": label, "Stiffness (N/m)": case.stiffness_n_per_m, "Peak tension (N)": out.peak_tension_n, "Maximum |deviation| (m)": out.maximum_abs_extension_m, "Settling time (s)": out.settling_time_s})
    return pd.DataFrame(rows)


def compare_damping(base: SimulationInputs) -> pd.DataFrame:
    center = base.damping_ns_per_m
    values = sorted({max(0.0, center * 0.5), center, center * 1.5})
    # If the nominal value is zero, still provide three meaningful test cases.
    if len(values) < 3:
        values = [0.0, 60.0, 120.0]
    labels = ["Low damping", "Nominal damping", "High damping"]
    rows = []
    for label, value in zip(labels, values):
        case = replace(base, damping_ns_per_m=value)
        out = run_simulation(case)
        rows.append({"Case": label, "Damping (N·s/m)": value, "Dissipated energy (J)": out.dissipated_energy_j, "Settling time (s)": out.settling_time_s, "Peak tension (N)": out.peak_tension_n})
    return pd.DataFrame(rows)
