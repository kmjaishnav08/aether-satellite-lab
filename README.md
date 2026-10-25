# AETHER — Satellite Capture & Tether Dynamics Virtual Lab

A Python/Streamlit educational virtual laboratory for a two-stage, one-dimensional model of satellite capture and post-capture tether dynamics.

## Features

- Interactive controls for spacecraft masses, closing speed, centre-of-mass velocity, tether stiffness, damping, nominal tension, and post-capture perturbation conditions.
- Perfectly inelastic capture calculation with pre/post momentum verification and collision energy-loss calculation.
- Numerical integration of a linearised post-capture spring–damper model.
- Animated 1-D schematic and time series for relative displacement deviation, relative extension rate, tether tension, stored oscillation energy, and dissipated energy.
- Controlled parameter studies for approach speed, stiffness, and damping.
- CSV export for the full time series and parameter summary.
- Automated physics tests.

## Windows setup

1. Install Python 3.11 or later from python.org and enable **Add Python to PATH** during installation.
2. Extract this folder to a location you can find, such as `Documents\Satellite_Tether_Lab`.
3. Open Command Prompt in the extracted folder.
4. Run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

5. Streamlit will print a local URL, usually `http://localhost:8501`. Open it in Chrome or Edge.
6. Use the sidebar to adjust one parameter at a time; use **Experiment comparison** to compare controlled cases.

## Physical model

### Stage 1: Idealised capture

The approach speed is the positive closing speed along one axis. With masses `mc` and `ms`, total mass `M`, and chosen centre-of-mass velocity `Vcm`, the model sets

- `vc_before = Vcm + (ms/M) * closing_speed`
- `vs_before = Vcm - (mc/M) * closing_speed`
- `V_after = (mc*vc_before + ms*vs_before)/M`

The perfectly inelastic collision energy loss is `0.5 * reduced_mass * closing_speed**2`, where `reduced_mass = mc*ms/M`.

### Stage 2: Small post-capture oscillations

The post-capture stage is a linearised one-dimensional perturbation about a nominal taut-tether operating point:

`reduced_mass * x_ddot + damping * x_dot + stiffness * x = 0`

The actual tension around the chosen operating point is represented by `T = T0 + stiffness*x + damping*x_dot`. The nominal force balance that maintains the operating point is not explicitly solved. If computed total tension becomes non-positive, the user is warned that the small-perturbation, taut-tether assumption is no longer valid for that case.

The initial post-capture extension deviation and extension rate are separate transient initial conditions; they are **not** interpreted as the relative velocity immediately after a perfectly inelastic collision. Collision energy loss is kept separate from post-capture damping dissipation.

## Scope and limitations

This model does **not** simulate a complete orbital rendezvous, 3-D attitude/tumbling, cable slack/contact, active grappling hardware, satellite repair, or a flight-certified system. It is a 1-D educational mechanics demonstrator intended to make assumptions measurable and explainable.

## Validation

The tests check momentum conservation, the inelastic collision energy-loss formula, the post-capture energy balance, zero-disturbance behaviour, positive tension for the standard demo case, and invalid-input rejection.
