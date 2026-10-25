"""Reusable Plotly visuals for the Satellite Capture & Tether Dynamics lab."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physics_model import SimulationResult

NAVY = "#0B1830"
TEAL = "#16B8B0"
BLUE = "#4D8DFF"
AMBER = "#F4B860"
RED = "#E15B64"
MUTED = "#718096"
GRID = "rgba(148,163,184,0.22)"


def _base_layout(fig: go.Figure, title: str, y_title: str, height: int = 300) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 16, "color": "#EAF1FF"}},
        height=height,
        margin=dict(l=12, r=15, t=54, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,24,47,0.38)",
        font=dict(family="Inter, Segoe UI, Arial", size=11, color="#D8E2F1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s)", gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def make_motion_animation(result: SimulationResult, visual_amplification: float = 12.0) -> go.Figure:
    """Create an animated 1-D schematic, with displacement visibly amplified."""
    df = result.data
    nframes = min(65, len(df))
    frame_indices = np.unique(np.linspace(0, len(df) - 1, nframes, dtype=int))
    mc = result.inputs.chaser_mass_kg
    mt = result.inputs.target_mass_kg
    total = mc + mt
    base_gap = 12.0

    def positions(idx: int) -> tuple[float, float]:
        # This is a schematic coordinate, not a physical orbital distance.
        gap = max(3.0, base_gap + visual_amplification * float(df.iloc[idx]["extension_perturbation_m"]))
        xc = -(mt / total) * gap
        xt = (mc / total) * gap
        return xc, xt

    x0c, x0t = positions(int(frame_indices[0]))
    y_craft = 0.0
    y_target = 0.0
    fig = go.Figure(
        data=[
            go.Scatter(x=[x0c, x0t], y=[0, 0], mode="lines", line=dict(color="#A8B8CC", width=5), name="Tether", hoverinfo="skip"),
            go.Scatter(x=[x0c], y=[y_craft], mode="markers+text", marker=dict(symbol="square", size=25, color=TEAL, line=dict(color="#D7FFF9", width=2)), text=["CHASER"], textposition="top center", name="Chaser", hovertemplate="Chaser spacecraft<extra></extra>"),
            go.Scatter(x=[x0t], y=[y_target], mode="markers+text", marker=dict(symbol="diamond", size=24, color=AMBER, line=dict(color="#FFF1D6", width=2)), text=["TARGET"], textposition="top center", name="Target satellite", hovertemplate="Target satellite<extra></extra>"),
        ]
    )
    frames = []
    for i, idx in enumerate(frame_indices):
        xc, xt = positions(int(idx))
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter(x=[xc, xt], y=[0, 0]),
                    go.Scatter(x=[xc], y=[y_craft]),
                    go.Scatter(x=[xt], y=[y_target]),
                ],
                traces=[0, 1, 2],
                layout=go.Layout(title_text=f"Post-capture tether response · t = {df.iloc[int(idx)]['time_s']:.2f} s"),
            )
        )
    fig.frames = frames
    fig.update_layout(
        title=dict(text=f"Post-capture motion · t = {df.iloc[0]['time_s']:.2f} s", x=0.02, xanchor="left", font=dict(size=16, color="#EAF1FF")),
        height=380,
        margin=dict(l=10, r=10, t=54, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,24,47,0.38)",
        font=dict(family="Inter, Segoe UI, Arial", size=11, color="#D8E2F1"),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showticklabels=False, title="Schematic axis (not to scale)"),
        yaxis=dict(visible=False, range=[-1.4, 1.4]),
        updatemenus=[
            dict(
                type="buttons", showactive=False, x=0.02, y=1.13, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 55, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Ⅱ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
                ],
            )
        ],
        sliders=[dict(
            active=0, x=0.16, y=1.13, len=0.82, currentvalue={"prefix": "Frame: ", "font": {"color": "#D8E2F1"}},
            steps=[dict(method="animate", args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label=f"{df.iloc[int(idx)]['time_s']:.1f}") for i, idx in enumerate(frame_indices)],
        )],
    )
    fig.add_annotation(
        text=f"Schematic only · relative displacement visually amplified {visual_amplification:.0f}×; charts use SI units",
        xref="paper", yref="paper", x=0.99, y=-0.12, showarrow=False, xanchor="right", font=dict(size=9, color="#9FB0C6"),
    )
    return fig


def make_extension_chart(result: SimulationResult) -> go.Figure:
    d = result.data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time_s, y=d.extension_perturbation_m, mode="lines", line=dict(color=TEAL, width=2.5), name="Extension deviation"))
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot", width=1))
    _base_layout(fig, "Relative displacement", "Deviation from nominal length (m)")
    return fig


def make_velocity_chart(result: SimulationResult) -> go.Figure:
    d = result.data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time_s, y=d.relative_extension_rate_mps, mode="lines", line=dict(color=BLUE, width=2.5), name="Relative rate"))
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot", width=1))
    _base_layout(fig, "Relative velocity", "Relative extension rate (m/s)")
    return fig


def make_tension_chart(result: SimulationResult) -> go.Figure:
    d = result.data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time_s, y=d.tether_tension_n, mode="lines", line=dict(color=AMBER, width=2.5), name="Tether tension"))
    fig.add_hline(y=result.inputs.nominal_tension_n, line=dict(color=MUTED, dash="dot", width=1), annotation_text="Nominal tension", annotation_position="top left")
    fig.add_hline(y=0, line=dict(color=RED, dash="dot", width=1))
    _base_layout(fig, "Tether tension vs time", "Tension (N)")
    return fig


def make_energy_chart(result: SimulationResult) -> go.Figure:
    d = result.data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time_s, y=d.oscillation_energy_j, mode="lines", line=dict(color=TEAL, width=2.5), name="Stored oscillation energy"))
    fig.add_trace(go.Scatter(x=d.time_s, y=d.dissipated_energy_j, mode="lines", line=dict(color=AMBER, width=2.5), name="Dissipated energy"))
    total = d.oscillation_energy_j + d.dissipated_energy_j
    fig.add_trace(go.Scatter(x=d.time_s, y=total, mode="lines", line=dict(color=BLUE, width=1.7, dash="dash"), name="Energy sum"))
    _base_layout(fig, "Energy accounting", "Energy (J)")
    return fig


def make_comparison_chart(labels: list[str], values: list[float], title: str, y_title: str, color: str = TEAL) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=color, text=[f"{v:,.3g}" for v in values], textposition="outside", cliponaxis=False))
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15, color="#EAF1FF")),
        height=310, margin=dict(l=12, r=12, t=54, b=34),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,24,47,0.38)",
        font=dict(family="Inter, Segoe UI, Arial", size=11, color="#D8E2F1"),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def make_spacecraft_animation_html(result: SimulationResult, visual_amplification: float = 12.0) -> str:
    """Return a self-contained HTML5 canvas animation using the real model data.

    The spacecraft are schematic vector drawings. Horizontal separation is
    visually amplified; the timeline/readouts and all graphs use the numerical
    simulation's SI-unit data. This is a 1-D visualisation, not an orbital solver.
    """
    import json

    df = result.data
    count = min(600, len(df))
    indices = np.unique(np.linspace(0, len(df) - 1, count, dtype=int))
    frames = []
    for idx in indices:
        row = df.iloc[int(idx)]
        frames.append({
            "t": float(row["time_s"]),
            "x": float(row["extension_perturbation_m"]),
            "xd": float(row["relative_extension_rate_mps"]),
            "T": float(row["tether_tension_n"]),
            "E": float(row["dissipated_energy_j"]),
        })
    payload = json.dumps(frames, separators=(",", ":"), allow_nan=False)
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:dark;--bg:#081426;--panel:#0c1d34;--line:#29415f;--text:#eaf1ff;--muted:#9aafca;--teal:#42d6c8;--blue:#66a5ff;--red:#ff777e;--amber:#f4c16e}
*{box-sizing:border-box}html,body{margin:0;padding:0;background:transparent;color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}
.shell{border:1px solid rgba(124,158,198,.30);border-radius:14px;overflow:hidden;background:linear-gradient(145deg,#0c1b31,#071222 75%);box-shadow:0 10px 28px rgba(0,0,0,.18)}
.head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:13px 16px;border-bottom:1px solid rgba(124,158,198,.18);background:rgba(16,37,64,.64)}
.kicker{font-size:11px;letter-spacing:.15em;font-weight:800;color:var(--teal);text-transform:uppercase}.state{font-size:10px;letter-spacing:.1em;font-weight:800;color:#a6bbd4;background:rgba(145,174,208,.08);border:1px solid rgba(145,174,208,.22);border-radius:999px;padding:5px 9px;white-space:nowrap}
.canvas-wrap{position:relative;padding:0 8px;background:#081426}canvas{display:block;width:100%;height:375px}
.readouts{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;padding:0 12px 11px}
.readout{min-width:0;border:1px solid rgba(124,158,198,.18);border-radius:9px;padding:8px 10px;background:rgba(13,31,54,.82)}
.readout small{display:block;color:var(--muted);font-size:9px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.readout strong{display:block;font-size:14px;line-height:1.3;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.controls{display:flex;align-items:center;gap:8px;padding:12px;border-top:1px solid rgba(124,158,198,.18);background:rgba(5,15,29,.7)}
button,select{font:600 12px Inter,"Segoe UI",Arial,sans-serif;color:var(--text);border:1px solid rgba(124,158,198,.32);background:#102641;border-radius:8px;padding:9px 11px;cursor:pointer}button.primary{background:#087e83;border-color:#24b9b0}button:hover,select:hover{border-color:var(--teal)}input[type=range]{flex:1;min-width:60px;accent-color:var(--teal);cursor:pointer}.time{font:11px ui-monospace,Consolas,monospace;color:#c3d2e5;white-space:nowrap}.foot{padding:0 14px 12px;color:#849bb8;font-size:10px;line-height:1.4}
@media(max-width:700px){canvas{height:320px}.readouts{grid-template-columns:repeat(3,minmax(0,1fr))}.controls{flex-wrap:wrap}.time{min-width:100%;text-align:right}}
</style>
</head>
<body>
<div class="shell">
 <div class="head"><div><div class="kicker">AETHER · Post-capture dynamics</div><div style="font-size:12px;color:#b8c9de;margin-top:4px">Live spacecraft-and-tether schematic</div></div><div class="state" id="stateBadge">PAUSED</div></div>
 <div class="canvas-wrap"><canvas id="scene" aria-label="Animated schematic of a chaser spacecraft connected to a target satellite by a tether"></canvas></div>
 <div class="readouts">
  <div class="readout"><small>Simulation time</small><strong id="vTime">0.00 s</strong></div>
  <div class="readout"><small>Extension deviation</small><strong id="vExt">0.000 m</strong></div>
  <div class="readout"><small>Relative rate</small><strong id="vRate">0.000 m/s</strong></div>
  <div class="readout"><small>Tether tension</small><strong id="vTension">0.0 N</strong></div>
  <div class="readout"><small>Damping loss</small><strong id="vEnergy">0.000 J</strong></div>
 </div>
 <div class="controls"><button class="primary" id="playBtn">▶ Play</button><button id="resetBtn">↺ Reset</button><input id="timeline" type="range" min="0" max="1" value="0" step="1" aria-label="Simulation timeline"><select id="speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><span class="time" id="timeLabel">0.00 s</span></div>
 <div class="foot">Schematic only: relative displacement is visually amplified by __GAIN__× for visibility. The displayed numerical readouts and separate scientific plots use calculated SI-unit data; spacecraft attitude and orbital trajectories are not simulated.</div>
</div>
<script>
const DATA = __DATA_JSON__;
const GAIN = __GAIN__;
const MC = __MC__;
const MT = __MT__;
const VCM = __VCM__;
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('timeline');
const playBtn = document.getElementById('playBtn');
const resetBtn = document.getElementById('resetBtn');
const speedSelect = document.getElementById('speed');
const badge = document.getElementById('stateBadge');
let index = 0, playing = false, lastStamp = null, lastRenderIndex = -1;
slider.max = String(DATA.length-1);
const clamp = (v,a,b)=>Math.max(a,Math.min(b,v));
function getState(i){const lo=Math.floor(clamp(i,0,DATA.length-1)), hi=Math.min(lo+1,DATA.length-1), f=clamp(i-lo,0,1);const a=DATA[lo],b=DATA[hi];return {t:a.t+(b.t-a.t)*f,x:a.x+(b.x-a.x)*f,xd:a.xd+(b.xd-a.xd)*f,T:a.T+(b.T-a.T)*f,E:a.E+(b.E-a.E)*f};}
function arrow(x1,y1,x2,y2,color,width=2){const ang=Math.atan2(y2-y1,x2-x1), head=7;ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-head*Math.cos(ang-.48),y2-head*Math.sin(ang-.48));ctx.lineTo(x2-head*Math.cos(ang+.48),y2-head*Math.sin(ang+.48));ctx.closePath();ctx.fill();ctx.restore();}
function label(text,x,y,color='#dce8f8',size=11,align='center'){ctx.save();ctx.fillStyle=color;ctx.font=`600 ${size}px Inter, Segoe UI, Arial`;ctx.textAlign=align;ctx.fillText(text,x,y);ctx.restore();}
function panel(x,y,w,h,s){ctx.save();ctx.fillStyle='#08254a';ctx.strokeStyle='#6798cd';ctx.lineWidth=1.1*s;ctx.fillRect(x-w/2,y-h/2,w,h);ctx.strokeRect(x-w/2,y-h/2,w,h);ctx.strokeStyle='rgba(92,159,223,.65)';ctx.lineWidth=.55*s;for(let xx=x-w/2+7*s;xx<x+w/2;xx+=7*s){ctx.beginPath();ctx.moveTo(xx,y-h/2);ctx.lineTo(xx,y+h/2);ctx.stroke()}for(let yy=y-h/2+8*s;yy<y+h/2;yy+=8*s){ctx.beginPath();ctx.moveTo(x-w/2,yy);ctx.lineTo(x+w/2,yy);ctx.stroke()}ctx.restore();}
function drawChaser(x,y,s){ctx.save();ctx.translate(x,y);ctx.scale(s,s);
 panel(-8,-69,42,63,1);panel(-8,69,42,63,1);
 // Rear engine bells
 for(let j=-1;j<=1;j++){ctx.fillStyle='#273d58';ctx.strokeStyle='#9cafc2';ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(-51,j*12,9,6,0,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#0a1523';ctx.beginPath();ctx.ellipse(-54,j*12,4,3,0,0,Math.PI*2);ctx.fill();}
 // Main cylindrical body with metal gradient
 const grad=ctx.createLinearGradient(0,-25,0,25);grad.addColorStop(0,'#e4edf5');grad.addColorStop(.35,'#8198ae');grad.addColorStop(.62,'#d5e0e9');grad.addColorStop(1,'#586f86');
 ctx.fillStyle=grad;ctx.strokeStyle='#d8e8f4';ctx.lineWidth=1.2;ctx.beginPath();ctx.roundRect(-43,-22,82,44,10);ctx.fill();ctx.stroke();
 ctx.fillStyle='#153d63';ctx.fillRect(-29,-22,11,44);ctx.fillRect(17,-22,9,44);
 ctx.strokeStyle='#516b83';ctx.lineWidth=1;for(let xx=-12;xx<17;xx+=8){ctx.beginPath();ctx.moveTo(xx,-17);ctx.lineTo(xx,17);ctx.stroke()}
 ctx.fillStyle='#22394f';ctx.fillRect(-7,-7,14,14);ctx.strokeStyle='#b5d8e5';ctx.strokeRect(-7,-7,14,14);
 ctx.fillStyle='#d7e2ec';ctx.beginPath();ctx.moveTo(39,-13);ctx.lineTo(51,-7);ctx.lineTo(51,7);ctx.lineTo(39,13);ctx.closePath();ctx.fill();ctx.strokeStyle='#94aabd';ctx.stroke();
 ctx.fillStyle='#c0cedc';ctx.beginPath();ctx.arc(51,0,6,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore();}
function drawTarget(x,y,s){ctx.save();ctx.translate(x,y);ctx.scale(s,s);
 // Irregular equipment bus
 const pts=[[-35,-24],[-20,-34],[13,-29],[34,-12],[32,20],[12,34],[-20,29],[-38,9]];
 const grad=ctx.createLinearGradient(-30,-30,30,30);grad.addColorStop(0,'#dce3e8');grad.addColorStop(.5,'#76899a');grad.addColorStop(1,'#35485c');ctx.fillStyle=grad;ctx.strokeStyle='#d5e1eb';ctx.lineWidth=1.2;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();ctx.stroke();
 // Gold restraint frame and panel seams
 ctx.strokeStyle='#c1a36b';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(-29,-23);ctx.lineTo(20,-28);ctx.lineTo(31,18);ctx.lineTo(-20,29);ctx.closePath();ctx.stroke();ctx.beginPath();ctx.moveTo(-20,-32);ctx.lineTo(-14,30);ctx.moveTo(15,-28);ctx.lineTo(18,32);ctx.moveTo(-36,1);ctx.lineTo(33,0);ctx.stroke();
 ctx.fillStyle='#152a40';ctx.beginPath();ctx.arc(-8,3,10,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#c9d4de';ctx.lineWidth=2;ctx.stroke();ctx.beginPath();ctx.arc(-8,3,5,0,Math.PI*2);ctx.stroke();
 // Solar panel on the upper-right side
 ctx.save();ctx.translate(44,-12);ctx.rotate(-.22);panel(0,0,27,51,1);ctx.restore();
 // Antenna booms
 ctx.strokeStyle='#c2ccd6';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(4,-28);ctx.lineTo(13,-47);ctx.lineTo(20,-51);ctx.moveTo(-26,23);ctx.lineTo(-38,38);ctx.stroke();ctx.fillStyle='#d9e0e5';ctx.beginPath();ctx.arc(20,-51,2.2,0,Math.PI*2);ctx.fill();ctx.restore();}
function draw(timeIndex){const box=canvas.getBoundingClientRect();const w=Math.max(320,box.width),h=Math.max(250,box.height),dpr=Math.min(window.devicePixelRatio||1,2);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);
 const bg=ctx.createLinearGradient(0,0,w,h);bg.addColorStop(0,'#0d2440');bg.addColorStop(.55,'#081729');bg.addColorStop(1,'#071322');ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
 // Subtle technical grid and fixed star points
 ctx.strokeStyle='rgba(113,150,190,.10)';ctx.lineWidth=1;for(let gx=20;gx<w;gx+=34){ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,h);ctx.stroke()}for(let gy=18;gy<h;gy+=34){ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(w,gy);ctx.stroke()}
 for(let i=0;i<46;i++){const sx=((i*173+41)%997)/997*w,sy=((i*97+19)%389)/389*h;ctx.fillStyle=`rgba(185,211,239,${.17+(i%4)*.08})`;ctx.beginPath();ctx.arc(sx,sy,(i%3===0)?1.2:.7,0,Math.PI*2);ctx.fill()}
 ctx.setLineDash([4,7]);ctx.strokeStyle='rgba(83,143,206,.38)';ctx.beginPath();ctx.ellipse(w*.5,h*.55,w*.38,h*.34,-.04,Math.PI*1.05,Math.PI*1.96);ctx.stroke();ctx.setLineDash([]);
 label('POST-CAPTURE TETHER RESPONSE',w/2,25,'#a7c1df',10,'center');
 const q=getState(timeIndex),M=MC+MT,baseGap=w*.46,delta=clamp(q.x*GAIN*35,-w*.11,w*.11),gap=baseGap+delta,cx=w*.5,cy=h*.51;
 const xc=cx-(MT/M)*gap,xd=cx+(MC/M)*gap,s=clamp(w/1000,.62,1.0); 
 // COM marker
 ctx.save();ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(66,214,200,.34)';ctx.beginPath();ctx.moveTo(cx,cy-107*s);ctx.lineTo(cx,cy+105*s);ctx.stroke();ctx.setLineDash([]);label('COM',cx,cy-117*s,'#65dacf',9);ctx.restore();
 // Tether behind craft, with very shallow sag for visibility
 const ax=xc+53*s,bx=xd-46*s,ty=cy+1*s;ctx.save();ctx.strokeStyle='rgba(66,214,200,.14)';ctx.lineWidth=7*s;ctx.beginPath();ctx.moveTo(ax,ty);ctx.quadraticCurveTo((ax+bx)/2,ty+6*s,bx,ty);ctx.stroke();ctx.strokeStyle='#b6c9dc';ctx.lineWidth=2.2*s;ctx.beginPath();ctx.moveTo(ax,ty);ctx.quadraticCurveTo((ax+bx)/2,ty+6*s,bx,ty);ctx.stroke();ctx.restore();
 drawChaser(xc,cy,s);drawTarget(xd,cy,s);
 // Inward tension forces acting at the two attachment points (Newton's third law)
 const fLen=clamp(25+Math.abs(q.T)*.015,28,56)*s,fy=cy+42*s;arrow(xc+51*s,fy,xc+51*s+fLen,fy,'#ff777e',2.2*s);arrow(xd-45*s,fy,xd-45*s-fLen,fy,'#ff777e',2.2*s);label('T(t)',xc+72*s,fy+18*s,'#ff9ca2',10);label('T(t)',xd-70*s,fy+18*s,'#ff9ca2',10);
 // Instantaneous velocities from the COM and relative-rate decomposition
 const vc=VCM-(MT/M)*q.xd,vd=VCM+(MC/M)*q.xd;const drawVelocity=(x,y,v,txt)=>{if(Math.abs(v)<.008){label(txt+' ≈ 0',x,y-55*s,'#a9c4e2',9);return;}const len=clamp(Math.abs(v)*100*s,15*s,66*s),end=x+Math.sign(v)*len;arrow(x,y,end,y,'#66a5ff',1.8*s);label(txt+' '+v.toFixed(2)+' m/s',(x+end)/2,y-10*s,'#9dc4ff',9);};drawVelocity(xc,cy-44*s,vc,'v_c');drawVelocity(xd,cy-44*s,vd,'v_d');
 // Small tether-state tag
 const msg=q.xd>0.006?'EXTENSION INCREASING':q.xd<-.006?'EXTENSION DECREASING':'NEAR TURNING POINT';ctx.fillStyle='rgba(7,18,32,.78)';ctx.strokeStyle='rgba(124,158,198,.28)';ctx.lineWidth=1;ctx.beginPath();ctx.roundRect(w/2-88, h-30, 176, 19, 8);ctx.fill();ctx.stroke();label(msg,w/2,h-17,q.xd>0.006?'#80c0ff':q.xd<-.006?'#ffbd91':'#9cdbc9',8);
 document.getElementById('vTime').textContent=q.t.toFixed(2)+' s';document.getElementById('vExt').textContent=(q.x*1000).toFixed(1)+' mm';document.getElementById('vRate').textContent=q.xd.toFixed(4)+' m/s';document.getElementById('vTension').textContent=q.T.toLocaleString(undefined,{maximumFractionDigits:1})+' N';document.getElementById('vEnergy').textContent=q.E.toFixed(3)+' J';document.getElementById('timeLabel').textContent=q.t.toFixed(2)+' s';
}
function render(){draw(index);slider.value=String(Math.round(index));}
function tick(stamp){if(!playing)return;if(lastStamp===null)lastStamp=stamp;const dt=Math.min((stamp-lastStamp)/1000,.08);lastStamp=stamp;const step=(DATA[DATA.length-1].t-DATA[0].t)/Math.max(1,DATA.length-1);index+=dt*Number(speedSelect.value)/step;if(index>=DATA.length-1){index=DATA.length-1;playing=false;playBtn.textContent='▶ Replay';badge.textContent='COMPLETE';lastStamp=null;}render();if(playing)requestAnimationFrame(tick);}
playBtn.addEventListener('click',()=>{if(playing){playing=false;lastStamp=null;playBtn.textContent='▶ Resume';badge.textContent='PAUSED';}else{if(index>=DATA.length-1)index=0;playing=true;lastStamp=null;playBtn.textContent='Ⅱ Pause';badge.textContent='PLAYING';requestAnimationFrame(tick);}render();});
resetBtn.addEventListener('click',()=>{playing=false;lastStamp=null;index=0;playBtn.textContent='▶ Play';badge.textContent='PAUSED';render();});
slider.addEventListener('input',()=>{playing=false;lastStamp=null;index=Number(slider.value);playBtn.textContent='▶ Play';badge.textContent='SCRUB';render();});
window.addEventListener('resize',()=>draw(index));render();
</script>
</body></html>'''
    return (template
            .replace("__DATA_JSON__", payload)
            .replace("__GAIN__", f"{float(visual_amplification):.6g}")
            .replace("__MC__", f"{float(result.inputs.chaser_mass_kg):.8g}")
            .replace("__MT__", f"{float(result.inputs.target_mass_kg):.8g}")
            .replace("__VCM__", f"{float(result.inputs.center_of_mass_velocity_mps):.8g}"))
