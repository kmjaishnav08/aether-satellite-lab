"""Physics engine for the Satellite Capture & Tether Dynamics Virtual Lab.

This is a deliberately simplified 1-D, two-stage educational model:
1) an idealised perfectly inelastic capture impulse, and
2) small post-capture oscillations about a nominal taut-tether operating point.

The second stage is a linearised spring-damper model. Nominal tether preload is
an operating-point reference; the force that drives perturbations is k*x + c*v.
The model does not solve orbital rendezvous, attitude motion, or cable slack/contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SimulationInputs:
    chaser_mass_kg: float = 500.0
    target_mass_kg: float = 200.0
    approach_speed_mps: float = 2.0
    center_of_mass_velocity_mps: float = 0.0
    stiffness_n_per_m: float = 500.0
    damping_ns_per_m: float = 120.0
    nominal_tension_n: float = 1000.0
    initial_extension_m: float = 0.15
    initial_extension_rate_mps: float = 0.15
    duration_s: float = 20.0
    samples: int = 1000

    def validate(self) -> None:
        positive = {
            "chaser_mass_kg": self.chaser_mass_kg,
            "target_mass_kg": self.target_mass_kg,
            "stiffness_n_per_m": self.stiffness_n_per_m,
            "nominal_tension_n": self.nominal_tension_n,
            "duration_s": self.duration_s,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be a finite value greater than zero.")
        nonnegative = {
            "approach_speed_mps": self.approach_speed_mps,
            "damping_ns_per_m": self.damping_ns_per_m,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative value.")
        signed = {
            "center_of_mass_velocity_mps": self.center_of_mass_velocity_mps,
            "initial_extension_m": self.initial_extension_m,
            "initial_extension_rate_mps": self.initial_extension_rate_mps,
        }
        for name, value in signed.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if abs(self.initial_extension_m) > 5:
            raise ValueError("Initial extension must remain within ±5 m for this small-perturbation model.")
        if abs(self.initial_extension_rate_mps) > 10:
            raise ValueError("Initial extension rate must remain within ±10 m/s for this model.")
        if int(self.samples) < 200 or int(self.samples) > 5000:
            raise ValueError("samples must be between 200 and 5000.")


@dataclass
class SimulationResult:
    inputs: SimulationInputs
    data: pd.DataFrame
    reduced_mass_kg: float
    total_mass_kg: float
    chaser_velocity_pre_mps: float
    target_velocity_pre_mps: float
    post_capture_velocity_mps: float
    momentum_before_kg_mps: float
    momentum_after_kg_mps: float
    momentum_error_kg_mps: float
    collision_energy_loss_j: float
    initial_oscillation_energy_j: float
    energy_balance_error_j: float
    peak_tension_n: float
    minimum_tension_n: float
    maximum_abs_extension_m: float
    maximum_abs_relative_rate_mps: float
    dissipated_energy_j: float
    settling_time_s: float | None
    natural_frequency_hz: float
    damping_ratio: float


def _settling_time(time_s: np.ndarray, x: np.ndarray, v: np.ndarray) -> float | None:
    """Return 2% settling time using displacement and velocity tolerances."""
    x_scale = max(float(np.max(np.abs(x))), 1e-12)
    v_scale = max(float(np.max(np.abs(v))), 1e-12)
    x_tol = max(0.02 * x_scale, 1e-5)
    v_tol = max(0.02 * v_scale, 1e-5)
    within = (np.abs(x) <= x_tol) & (np.abs(v) <= v_tol)
    if within.all():
        return 0.0
    # Find the first index after which every subsequent sample remains in-band.
    suffix_ok = np.logical_and.accumulate(within[::-1])[::-1]
    indices = np.flatnonzero(suffix_ok)
    return float(time_s[indices[0]]) if len(indices) else None


def run_simulation(inputs: SimulationInputs) -> SimulationResult:
    """Run the two-stage capture and post-capture tether perturbation model."""
    inputs.validate()

    mc = float(inputs.chaser_mass_kg)
    mt = float(inputs.target_mass_kg)
    total_mass = mc + mt
    reduced_mass = mc * mt / total_mass
    v_close = float(inputs.approach_speed_mps)
    v_cm = float(inputs.center_of_mass_velocity_mps)

    # Choose pre-capture velocities that yield the selected closing speed and
    # centre-of-mass velocity in a common one-dimensional reference frame.
    v_chaser_pre = v_cm + (mt / total_mass) * v_close
    v_target_pre = v_cm - (mc / total_mass) * v_close
    momentum_before = mc * v_chaser_pre + mt * v_target_pre
    v_post = momentum_before / total_mass
    momentum_after = total_mass * v_post
    momentum_error = momentum_after - momentum_before

    # For a perfectly inelastic capture, loss of relative kinetic energy is
    # 1/2 * reduced_mass * (relative closing speed)^2.
    collision_loss = 0.5 * reduced_mass * v_close**2

    k = float(inputs.stiffness_n_per_m)
    c = float(inputs.damping_ns_per_m)
    x0 = float(inputs.initial_extension_m)
    v0 = float(inputs.initial_extension_rate_mps)
    duration = float(inputs.duration_s)
    time = np.linspace(0.0, duration, int(inputs.samples))
    initial_osc_energy = 0.5 * reduced_mass * v0**2 + 0.5 * k * x0**2

    # Perturbation dynamics about a nominal taut-tether operating point:
    # mu*x'' + c*x' + k*x = 0. The operating point's steady balance is not
    # solved explicitly; only small axial deviations from it are simulated.
    def rhs(_t: float, state: np.ndarray) -> tuple[float, float, float]:
        x, xdot, _e_diss = state
        xddot = -(k * x + c * xdot) / reduced_mass
        power_diss = c * xdot**2
        return xdot, xddot, power_diss

    omega_n = math.sqrt(k / reduced_mass)
    max_step = min(duration / 200.0, 0.05 / max(omega_n, 1e-9))
    solution = solve_ivp(
        rhs,
        (0.0, duration),
        (x0, v0, 0.0),
        t_eval=time,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(f"Numerical integration failed: {solution.message}")

    x = solution.y[0]
    xdot = solution.y[1]
    e_diss = np.maximum(solution.y[2], 0.0)
    force_increment = k * x + c * xdot
    tension = float(inputs.nominal_tension_n) + force_increment
    spring_energy = 0.5 * k * x**2
    relative_kinetic_energy = 0.5 * reduced_mass * xdot**2
    oscillation_energy = spring_energy + relative_kinetic_energy
    energy_error = float(np.max(np.abs(oscillation_energy + e_diss - initial_osc_energy)))

    data = pd.DataFrame(
        {
            "time_s": time,
            "extension_perturbation_m": x,
            "relative_extension_rate_mps": xdot,
            "tether_force_increment_n": force_increment,
            "tether_tension_n": tension,
            "spring_energy_j": spring_energy,
            "relative_kinetic_energy_j": relative_kinetic_energy,
            "oscillation_energy_j": oscillation_energy,
            "dissipated_energy_j": e_diss,
        }
    )

    zeta = c / (2.0 * math.sqrt(k * reduced_mass)) if c > 0 else 0.0
    omega_d = omega_n * math.sqrt(max(0.0, 1.0 - zeta**2)) if zeta < 1.0 else 0.0
    natural_frequency_hz = omega_d / (2.0 * math.pi) if omega_d > 0 else 0.0

    return SimulationResult(
        inputs=inputs,
        data=data,
        reduced_mass_kg=reduced_mass,
        total_mass_kg=total_mass,
        chaser_velocity_pre_mps=v_chaser_pre,
        target_velocity_pre_mps=v_target_pre,
        post_capture_velocity_mps=v_post,
        momentum_before_kg_mps=momentum_before,
        momentum_after_kg_mps=momentum_after,
        momentum_error_kg_mps=momentum_error,
        collision_energy_loss_j=collision_loss,
        initial_oscillation_energy_j=initial_osc_energy,
        energy_balance_error_j=energy_error,
        peak_tension_n=float(np.max(tension)),
        minimum_tension_n=float(np.min(tension)),
        maximum_abs_extension_m=float(np.max(np.abs(x))),
        maximum_abs_relative_rate_mps=float(np.max(np.abs(xdot))),
        dissipated_energy_j=float(e_diss[-1]),
        settling_time_s=_settling_time(time, x, xdot),
        natural_frequency_hz=natural_frequency_hz,
        damping_ratio=zeta,
    )


def validation_summary(result: SimulationResult) -> dict[str, bool | float]:
    """Return transparent checks that the UI can display to the user."""
    return {
        "momentum_conserved": abs(result.momentum_error_kg_mps) < 1e-7,
        "energy_balance_within_tolerance": result.energy_balance_error_j < max(1e-5, result.initial_oscillation_energy_j * 1e-5),
        "tether_remains_taut": result.minimum_tension_n > 0.0,
        "momentum_error_kg_mps": abs(result.momentum_error_kg_mps),
        "energy_balance_error_j": result.energy_balance_error_j,
    }
