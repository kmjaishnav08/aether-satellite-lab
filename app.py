from __future__ import annotations

from dataclasses import asdict
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from physics_model import SimulationInputs, run_simulation, validation_summary
from experiment_data import compare_approach_speed, compare_stiffness, compare_damping
from visualisation import (
    make_spacecraft_animation_html,
    make_extension_chart,
    make_velocity_chart,
    make_tension_chart,
    make_energy_chart,
    make_comparison_chart,
    TEAL, BLUE, AMBER,
)

st.set_page_config(
    page_title="AETHER | Satellite Capture & Tether Dynamics Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { background: radial-gradient(circle at 10% 0%, #15294a 0, #081426 43%, #06101f 100%); color: #eaf1ff; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0c1c33 0%, #091527 100%); border-right: 1px solid rgba(142,170,205,.18); }
    h1,h2,h3,p,span,label,div { font-family: Inter, Segoe UI, Arial, sans-serif; }
    h1,h2,h3 { letter-spacing: -0.025em; }
    .hero-kicker { color:#48d6c8; font-size:.72rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.4rem; }
    .hero-title { color:#f3f7ff; font-size:2rem; line-height:1.12; font-weight:800; margin:0; }
    .hero-sub { color:#a9bbd2; font-size:.96rem; margin-top:.65rem; max-width:900px; }
    .creator-credit { color:#c7d7e9; font-size:.83rem; margin-top:.55rem; letter-spacing:.02em; }
    .creator-credit strong { color:#63d8cd; font-weight:800; }
    .creator-watermark { position:fixed; right:18px; bottom:12px; z-index:0; pointer-events:none; opacity:.045; color:#b9d8f2; font-size:1.15rem; font-weight:800; letter-spacing:.18em; transform:rotate(-8deg); }
    .section-label { color:#63d8cd; font-size:.72rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.3rem; }
    .panel { background:rgba(13,30,54,.76); border:1px solid rgba(145,174,208,.20); border-radius:15px; padding:16px 18px; }
    .small-note { font-size:.78rem; color:#9caec5; line-height:1.5; }
    .status-pill { display:inline-block; border-radius:999px; padding:5px 10px; font-size:.72rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; background:rgba(22,184,176,.12); color:#64e0d4; border:1px solid rgba(22,184,176,.32); }
    div[data-testid="stMetric"] { background:rgba(13,30,54,.72); border:1px solid rgba(145,174,208,.2); border-radius:13px; padding:13px 15px; }
    div[data-testid="stMetricLabel"] { color:#a9bbd2 !important; }
    div[data-testid="stMetricValue"] { color:#f3f7ff !important; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid rgba(145,174,208,.18); }
    .stTabs [data-baseweb="tab"] { background:rgba(13,30,54,.6); border-radius:10px 10px 0 0; padding:10px 16px; color:#b6c6db; }
    .stTabs [aria-selected="true"] { color:#63d8cd !important; background:rgba(22,184,176,.10) !important; }
    div.stButton > button, div.stDownloadButton > button { border-radius:9px; border:1px solid rgba(99,216,205,.35); background:#127f85; color:white; font-weight:700; }
    div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#63d8cd; color:white; }
    hr { border-color:rgba(145,174,208,.17); }
    </style>
    """,
    unsafe_allow_html=True,
)

PROFILES = {
    "Standard demonstration": {"chaser_mass_kg": 500.0, "target_mass_kg": 200.0, "approach_speed_mps": 2.0, "center_of_mass_velocity_mps": 0.0, "stiffness_n_per_m": 500.0, "damping_ns_per_m": 120.0, "nominal_tension_n": 1000.0, "initial_extension_m": 0.15, "initial_extension_rate_mps": 0.15, "duration_s": 20.0},
    "Low damping study": {"chaser_mass_kg": 500.0, "target_mass_kg": 200.0, "approach_speed_mps": 2.0, "center_of_mass_velocity_mps": 0.0, "stiffness_n_per_m": 500.0, "damping_ns_per_m": 35.0, "nominal_tension_n": 1000.0, "initial_extension_m": 0.15, "initial_extension_rate_mps": 0.15, "duration_s": 20.0},
    "High damping study": {"chaser_mass_kg": 500.0, "target_mass_kg": 200.0, "approach_speed_mps": 2.0, "center_of_mass_velocity_mps": 0.0, "stiffness_n_per_m": 500.0, "damping_ns_per_m": 320.0, "nominal_tension_n": 1000.0, "initial_extension_m": 0.15, "initial_extension_rate_mps": 0.15, "duration_s": 20.0},
    "Stiffer tether study": {"chaser_mass_kg": 500.0, "target_mass_kg": 200.0, "approach_speed_mps": 2.0, "center_of_mass_velocity_mps": 0.0, "stiffness_n_per_m": 1400.0, "damping_ns_per_m": 120.0, "nominal_tension_n": 1000.0, "initial_extension_m": 0.15, "initial_extension_rate_mps": 0.15, "duration_s": 20.0},
    "Higher approach-speed study": {"chaser_mass_kg": 500.0, "target_mass_kg": 200.0, "approach_speed_mps": 4.0, "center_of_mass_velocity_mps": 0.0, "stiffness_n_per_m": 500.0, "damping_ns_per_m": 120.0, "nominal_tension_n": 1000.0, "initial_extension_m": 0.15, "initial_extension_rate_mps": 0.15, "duration_s": 20.0},
}

st.sidebar.markdown('<div class="section-label">Mission setup</div>', unsafe_allow_html=True)
profile_name = st.sidebar.selectbox("Experiment preset", list(PROFILES.keys()), key="profile_select")
if st.session_state.get("_last_profile") != profile_name:
    for name, value in PROFILES[profile_name].items():
        st.session_state[f"p_{name}"] = value
    st.session_state["_last_profile"] = profile_name

st.sidebar.markdown("### 01 · Spacecraft properties")
mc = st.sidebar.number_input("Chaser mass (kg)", min_value=10.0, max_value=10000.0, step=10.0, key="p_chaser_mass_kg")
mt = st.sidebar.number_input("Target satellite mass (kg)", min_value=10.0, max_value=10000.0, step=10.0, key="p_target_mass_kg")
vclose = st.sidebar.number_input("Relative approach speed (m/s)", min_value=0.0, max_value=20.0, step=0.1, key="p_approach_speed_mps")
vcm = st.sidebar.number_input("Centre-of-mass velocity (m/s)", min_value=-20.0, max_value=20.0, step=0.1, key="p_center_of_mass_velocity_mps", help="Velocity in the selected one-dimensional reference frame; set zero to use the centre-of-mass frame.")

st.sidebar.markdown("### 02 · Tether operating point")
k = st.sidebar.number_input("Axial stiffness (N/m)", min_value=10.0, max_value=5000.0, step=10.0, key="p_stiffness_n_per_m")
c = st.sidebar.number_input("Damping coefficient (N·s/m)", min_value=0.0, max_value=1000.0, step=5.0, key="p_damping_ns_per_m")
t0 = st.sidebar.number_input("Nominal tether tension (N)", min_value=100.0, max_value=10000.0, step=50.0, key="p_nominal_tension_n", help="Reference tension around which the small-perturbation model is linearised.")
x0 = st.sidebar.number_input("Initial extension deviation (m)", min_value=-0.5, max_value=0.5, step=0.01, key="p_initial_extension_m")
xdot0 = st.sidebar.number_input("Initial extension rate (m/s)", min_value=-1.0, max_value=1.0, step=0.01, key="p_initial_extension_rate_mps")

st.sidebar.markdown("### 03 · Solver and display")
duration = st.sidebar.slider("Simulation duration (s)", min_value=2.0, max_value=60.0, step=1.0, key="p_duration_s")
samples = st.sidebar.select_slider("Output resolution", options=[400, 600, 800, 1000, 1400, 1800], value=1000)
visual_gain = st.sidebar.slider("Animation displacement amplification", min_value=4, max_value=24, value=12, help="Only the schematic is visually amplified. Graphs remain in SI units.")
st.sidebar.caption("All calculations use SI units. Presets can be changed and then adjusted manually.")

inputs = SimulationInputs(
    chaser_mass_kg=mc,
    target_mass_kg=mt,
    approach_speed_mps=vclose,
    center_of_mass_velocity_mps=vcm,
    stiffness_n_per_m=k,
    damping_ns_per_m=c,
    nominal_tension_n=t0,
    initial_extension_m=x0,
    initial_extension_rate_mps=xdot0,
    duration_s=duration,
    samples=samples,
)

try:
    result = run_simulation(inputs)
except (ValueError, RuntimeError) as exc:
    st.error(f"The model could not run with these inputs: {exc}")
    st.stop()

st.markdown('<div class="hero-kicker">Computational aerospace mechanics · virtual laboratory</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">AETHER <span style="color:#63d8cd; font-weight:500;">/ Satellite Capture & Tether Dynamics</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">A reproducible 1-D study of capture momentum and post-capture oscillations around a nominal taut-tether operating point.</div>', unsafe_allow_html=True)
st.markdown('<div class="creator-credit">Developed by <strong>Jaishnav K M</strong> &nbsp;·&nbsp; Aerospace Engineering Student &amp; Project Developer</div>', unsafe_allow_html=True)
st.markdown('<div class="creator-watermark">JAISHNAV K M · AEROSPACE</div>', unsafe_allow_html=True)
st.write("")
validation = validation_summary(result)
if validation["tether_remains_taut"]:
    st.markdown('<span class="status-pill">Model regime · taut reference state</span>', unsafe_allow_html=True)
else:
    st.warning("The linearised model predicts non-positive total tension during this run. The small-perturbation, taut-tether assumption is no longer valid; increase nominal tension or reduce the initial perturbation.")
st.write("")

# KPI row
kpi = st.columns(5)
kpi[0].metric("COM velocity (m/s)", f"{result.post_capture_velocity_mps:.3f}")
kpi[1].metric("Collision loss (kJ)", f"{result.collision_energy_loss_j / 1000:.3f}")
kpi[2].metric("Peak tension (N)", f"{result.peak_tension_n:,.1f}")
kpi[3].metric("Peak deviation (mm)", f"{result.maximum_abs_extension_m * 1000:.1f}")
kpi[4].metric("Damping loss (J)", f"{result.dissipated_energy_j:.3f}")

mission_tab, compare_tab, physics_tab, export_tab = st.tabs(["◈ Mission dashboard", "⇄ Experiment comparison", "∑ Physics & assumptions", "↓ Data & validation"])

with mission_tab:
    st.markdown('<div class="section-label">Live spacecraft animation</div>', unsafe_allow_html=True)
    components.html(
        make_spacecraft_animation_html(result, visual_amplification=float(visual_gain)),
        height=560,
        scrolling=False,
    )
    st.caption("The chaser and target are drawn as spacecraft, and their separation, velocity arrows, tether force and readouts update from the calculated post-capture time history. Displacement is visually amplified; numerical values and scientific graphs remain in SI units.")

    left, right = st.columns([1.04, 1.0], gap="large")
    with left:
        st.markdown('<div class="section-label">Capture event · momentum check</div>', unsafe_allow_html=True)
        collision_table = pd.DataFrame([
            {"Quantity": "Chaser velocity before capture", "Value": f"{result.chaser_velocity_pre_mps:.4f} m/s"},
            {"Quantity": "Target velocity before capture", "Value": f"{result.target_velocity_pre_mps:.4f} m/s"},
            {"Quantity": "Common velocity after idealised capture", "Value": f"{result.post_capture_velocity_mps:.4f} m/s"},
            {"Quantity": "Total momentum before", "Value": f"{result.momentum_before_kg_mps:.6f} kg·m/s"},
            {"Quantity": "Total momentum after", "Value": f"{result.momentum_after_kg_mps:.6f} kg·m/s"},
        ])
        st.dataframe(collision_table, hide_index=True, use_container_width=True)
        if validation["momentum_conserved"]:
            st.success(f"Momentum check passed · absolute numerical error {validation['momentum_error_kg_mps']:.2e} kg·m/s")
        else:
            st.error("Momentum validation failed. Review the input case and numerical calculations.")
    with right:
        st.markdown('<div class="section-label">Tether response</div>', unsafe_allow_html=True)
        st.plotly_chart(make_tension_chart(result), use_container_width=True, config={"displaylogo": False})

    row_a, row_b = st.columns(2, gap="large")
    with row_a:
        st.plotly_chart(make_extension_chart(result), use_container_width=True, config={"displaylogo": False})
        st.plotly_chart(make_velocity_chart(result), use_container_width=True, config={"displaylogo": False})
    with row_b:
        st.plotly_chart(make_energy_chart(result), use_container_width=True, config={"displaylogo": False})
        st.markdown('<div class="panel"><div class="section-label">Transient characteristics</div>', unsafe_allow_html=True)
        settle = "Not settled in selected duration" if result.settling_time_s is None else f"{result.settling_time_s:.2f} s"
        st.metric("Estimated 2% settling time", settle)
        st.metric("Damping ratio", f"{result.damping_ratio:.3f}")
        st.metric("Damped natural frequency", f"{result.natural_frequency_hz:.3f} Hz")
        st.markdown('<div class="small-note">Settling time is estimated from the displacement and relative-rate traces using a 2% threshold. For zero initial disturbance, settling time is zero.</div></div>', unsafe_allow_html=True)

with compare_tab:
    st.markdown('<div class="section-label">Controlled parameter studies</div>', unsafe_allow_html=True)
    study = st.selectbox("Choose a study", ["Approach speed → collision energy", "Stiffness → tether response", "Damping → settling and energy"], key="comparison_study")
    if study.startswith("Approach"):
        comparison = compare_approach_speed(inputs)
        st.markdown("**Question:** How does the relative approach speed affect energy lost during the idealised inelastic capture? The post-capture initial disturbance is held constant for a fair, two-stage comparison.")
        st.plotly_chart(make_comparison_chart(comparison["Case"].tolist(), comparison["Collision energy loss (J)"].tolist(), "Collision energy loss by approach-speed case", "Energy lost (J)", BLUE), use_container_width=True)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    elif study.startswith("Stiffness"):
        comparison = compare_stiffness(inputs)
        st.markdown("**Question:** How does stiffness change the peak response? Masses, damping and initial disturbance are held constant.")
        st.plotly_chart(make_comparison_chart(comparison["Case"].tolist(), comparison["Peak tension (N)"].tolist(), "Peak tension by stiffness case", "Peak tension (N)", TEAL), use_container_width=True)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    else:
        comparison = compare_damping(inputs)
        st.markdown("**Question:** How does damping affect settling time and the amount of vibration energy dissipated? Masses, stiffness and initial disturbance are held constant.")
        c1, c2 = st.columns(2)
        with c1:
            vals = comparison["Dissipated energy (J)"].fillna(0.0).tolist()
            st.plotly_chart(make_comparison_chart(comparison["Case"].tolist(), vals, "Damping energy comparison", "Dissipated energy (J)", AMBER), use_container_width=True)
        with c2:
            settle_values = [v if pd.notna(v) else 0.0 for v in comparison["Settling time (s)"].tolist()]
            st.plotly_chart(make_comparison_chart(comparison["Case"].tolist(), settle_values, "Settling-time comparison", "Settling time (s)", BLUE), use_container_width=True)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption("Low/current/high cases are calculated by rerunning the same physics engine with one controlled parameter changed.")

with physics_tab:
    st.markdown("## Model equations")
    st.latex(r"v_{cm}=\frac{m_c v_c+m_s v_s}{m_c+m_s}")
    st.write("The capture stage uses a one-dimensional perfectly inelastic-collision idealisation. The chosen pre-capture velocities are constructed from the user-entered closing speed and centre-of-mass velocity so that total momentum is conserved.")
    st.latex(r"\mu=\frac{m_c m_s}{m_c+m_s}")
    st.latex(r"\mu\ddot{x}+c\dot{x}+kx=0")
    st.write("The post-capture stage models small axial oscillations around a nominal taut-tether operating point. Here, x is the deviation from that nominal separation; it is not the absolute tether length.")
    st.latex(r"T(t)=T_0+kx+c\dot{x}")
    st.latex(r"E_{diss}(t)=\int_0^t c\dot{x}^{\,2}\,dt")
    st.markdown("### Assumptions and limitations")
    st.markdown("""
    - One-dimensional translational dynamics; no full orbital trajectory or 3-D attitude/tumbling model.
    - The capture impulse is idealised as perfectly inelastic; external impulse during the short event is neglected.
    - The post-capture response is a linearised spring–damper perturbation about a nominal taut operating point. The operating point's steady force balance is not solved explicitly.
    - The initial post-capture extension and extension rate are separate transient initial conditions. They are **not** assumed to be the immediate relative velocity after a perfectly inelastic collision.
    - If total tension falls to zero or below, the linearised taut-tether assumption is invalid for that run; use a smaller disturbance or larger nominal tension.
    - Energy lost during the idealised collision is reported separately from energy dissipated by the post-capture damper.
    - This is an educational mechanics model, not a flight-certified satellite servicing or orbital-navigation system.
    """)
    st.info("These explicit assumptions are a strength during a viva: they show exactly what the simulation does and what it does not claim to model.")
    st.image("conceptual_capture_figure.png", caption="Conceptual reference illustration; the animated schematic and all plotted values are generated from the numerical model.", use_container_width=True)

with export_tab:
    st.markdown('<div class="section-label">Reproducible experiment record</div>', unsafe_allow_html=True)
    st.write("Export the full time history together with the current inputs and numerical validation results.")
    raw_csv = result.data.to_csv(index=False).encode("utf-8")
    st.download_button("Download time-series CSV", data=raw_csv, file_name="aether_tether_simulation.csv", mime="text/csv")
    summary = {
        **asdict(inputs),
        "reduced_mass_kg": result.reduced_mass_kg,
        "pre_capture_chaser_velocity_mps": result.chaser_velocity_pre_mps,
        "pre_capture_target_velocity_mps": result.target_velocity_pre_mps,
        "post_capture_com_velocity_mps": result.post_capture_velocity_mps,
        "momentum_before_kg_mps": result.momentum_before_kg_mps,
        "momentum_after_kg_mps": result.momentum_after_kg_mps,
        "momentum_error_kg_mps": result.momentum_error_kg_mps,
        "collision_energy_loss_j": result.collision_energy_loss_j,
        "initial_oscillation_energy_j": result.initial_oscillation_energy_j,
        "damper_energy_dissipated_j": result.dissipated_energy_j,
        "energy_balance_error_j": result.energy_balance_error_j,
        "peak_tension_n": result.peak_tension_n,
        "minimum_tension_n": result.minimum_tension_n,
        "max_abs_extension_deviation_m": result.maximum_abs_extension_m,
        "estimated_settling_time_s": result.settling_time_s,
        "natural_frequency_hz": result.natural_frequency_hz,
        "damping_ratio": result.damping_ratio,
    }
    summary_csv = pd.DataFrame([summary]).to_csv(index=False).encode("utf-8")
    st.download_button("Download experiment summary", data=summary_csv, file_name="aether_experiment_summary.csv", mime="text/csv")
    checks = validation_summary(result)
    v1, v2 = st.columns(2)
    with v1:
        st.write("**Momentum conservation**")
        st.success("PASS" if checks["momentum_conserved"] else "CHECK")
        st.caption(f"Absolute momentum error: {checks['momentum_error_kg_mps']:.3e} kg·m/s")
    with v2:
        st.write("**Post-capture energy balance**")
        st.success("PASS" if checks["energy_balance_within_tolerance"] else "CHECK")
        st.caption(f"Maximum energy-balance residual: {checks['energy_balance_error_j']:.3e} J")
    st.markdown("### Current parameter set")
    st.dataframe(pd.DataFrame([asdict(inputs)]).T.rename(columns={0: "Value"}), use_container_width=True)

st.divider()
st.markdown('<div class="small-note">AETHER · Developed by Jaishnav K M · Aerospace Engineering Student · Educational numerical model.</div>', unsafe_allow_html=True)
