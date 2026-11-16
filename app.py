"""Interactive simulation of the three-cell flotation level-control case.

The implementation follows the mathematical relationships reported in:
Quintanilla et al., MethodsX 10 (2023) 102081, doi:10.1016/j.mex.2023.102081.

Digital Twin & SCADA Interface with interactive P&ID, feedback-feedforward
decoupling, and comparative performance analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.set_page_config(
    page_title="Control de Nivel en Flotación | SCADA Digital Twin",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CELL_NAMES = ("Celda 1", "Celda 2", "Celda 3")
CELL_COLORS = ("#06B6D4", "#F59E0B", "#8B5CF6")
CELL_BG_COLORS = ("rgba(6, 182, 212, 0.15)", "rgba(245, 158, 11, 0.15)", "rgba(139, 92, 246, 0.15)")


@dataclass(frozen=True)
class ArticleParameters:
    """Parameters transcribed from Table 2 and Eq. (7) of the article."""

    k_integrating: np.ndarray
    delay_s: np.ndarray
    kc: np.ndarray
    ti_s: np.ndarray
    kff: np.ndarray


ARTICLE = ArticleParameters(
    k_integrating=np.array([-0.001018, -0.001018, -0.001018], dtype=float),
    delay_s=np.array([7.0, 6.0, 3.0], dtype=float),
    kc=np.array([-62.0, -76.0, -85.0], dtype=float),
    ti_s=np.array([56.0, 30.0, 44.0], dtype=float),
    kff=np.array([0.8, 0.8, 1.0], dtype=float),
)


@dataclass
class SimulationResult:
    time_s: np.ndarray
    levels_cm: np.ndarray
    flows_lpm: np.ndarray
    disturbances_lpm: np.ndarray
    errors_cm: np.ndarray
    cumulative_iae: np.ndarray
    iae_cm_s: np.ndarray
    sse_cm2_s: np.ndarray
    peak_errors_cm: np.ndarray
    settling_times_s: np.ndarray


def _as_three(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (3,):
        raise ValueError("Se requieren exactamente tres valores, uno por celda.")
    return array


def simulate(
    *,
    duration_s: float,
    dt_s: float,
    setpoints_cm: Iterable[float],
    initial_levels_cm: Iterable[float],
    nominal_flow_lpm: float,
    disturbance_start_s: float,
    disturbance_step_lpm: float,
    use_feedforward: bool,
    parameters: ArticleParameters = ARTICLE,
    flow_min_lpm: float = 0.0,
    flow_max_lpm: float = 100.0,
) -> SimulationResult:
    """Simulate three delayed integrating cells in series.

    The disturbance of cell 1 is the external feed flow. The disturbances of
    cells 2 and 3 are the previous cell's tailings flow, as in Fig. 15 of the
    article. The manipulated variables are the three tailings flows.
    """

    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("La duración y el paso de simulación deben ser positivos.")
    if flow_min_lpm >= flow_max_lpm:
        raise ValueError("El límite inferior de caudal debe ser menor al superior.")

    setpoints = _as_three(setpoints_cm)
    initial_levels = _as_three(initial_levels_cm)
    n_steps = int(np.floor(duration_s / dt_s)) + 1
    time_s = np.arange(n_steps, dtype=float) * dt_s

    levels = np.zeros((n_steps, 3), dtype=float)
    flows = np.zeros((n_steps, 3), dtype=float)
    disturbances = np.zeros((n_steps, 3), dtype=float)
    errors = np.zeros((n_steps, 3), dtype=float)
    levels[0] = initial_levels

    nominal = float(nominal_flow_lpm)
    integral_error = np.zeros(3, dtype=float)
    delay_steps = np.maximum(0, np.rint(parameters.delay_s / dt_s).astype(int))

    for k in range(n_steps - 1):
        disturbances[k, 0] = nominal + (
            disturbance_step_lpm if time_s[k] >= disturbance_start_s else 0.0
        )
        disturbances[k, 1] = flows[k - 1, 0] if k > 0 else nominal
        disturbances[k, 2] = flows[k - 1, 1] if k > 0 else nominal

        errors[k] = setpoints - levels[k]
        integral_error += errors[k] * dt_s
        pi_term = parameters.kc * (errors[k] + integral_error / parameters.ti_s)

        # Feedforward compensation on deviation variable: Delta Qd = Qd - Qd,nominal
        delta_disturbances = disturbances[k] - nominal
        feedforward_term = parameters.kff * delta_disturbances if use_feedforward else 0.0

        # Eq. (7) from the article plus nominal-flow bias
        flows[k] = np.clip(
            nominal + pi_term + feedforward_term,
            flow_min_lpm,
            flow_max_lpm,
        )

        delayed_flows = np.array(
            [
                nominal
                if k - delay_steps[i] < 0
                else flows[k - delay_steps[i], i]
                for i in range(3)
            ],
            dtype=float,
        )

        # Integrating model Gp(s) = k' exp(-theta s) / s
        level_derivative = parameters.k_integrating * (delayed_flows - disturbances[k])
        levels[k + 1] = np.clip(levels[k] + level_derivative * dt_s, 0.0, 39.0)

    disturbances[-1, 0] = nominal + (
        disturbance_step_lpm if time_s[-1] >= disturbance_start_s else 0.0
    )
    disturbances[-1, 1] = flows[-2, 0] if n_steps > 1 else nominal
    disturbances[-1, 2] = flows[-2, 1] if n_steps > 1 else nominal
    flows[-1] = flows[-2] if n_steps > 1 else nominal
    errors[-1] = setpoints - levels[-1]

    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    iae = integrate(np.abs(errors), time_s, axis=0)
    sse = integrate(errors**2, time_s, axis=0)
    cumulative_iae = np.cumsum(np.abs(errors) * dt_s, axis=0)
    peak_errors = np.max(np.abs(errors), axis=0)

    # Settling time calculation after disturbance (tolerance +/- 0.25 cm)
    settling_times = np.full(3, np.nan, dtype=float)
    dist_idx = int(np.searchsorted(time_s, disturbance_start_s))
    tolerance = 0.25
    for i in range(3):
        post_err = np.abs(errors[dist_idx:, i])
        outside = np.where(post_err > tolerance)[0]
        if len(outside) == 0:
            settling_times[i] = 0.0
        elif outside[-1] + 1 < len(post_err):
            settling_times[i] = (outside[-1] + 1) * dt_s
        else:
            settling_times[i] = time_s[-1] - disturbance_start_s

    return SimulationResult(
        time_s=time_s,
        levels_cm=levels,
        flows_lpm=flows,
        disturbances_lpm=disturbances,
        errors_cm=errors,
        cumulative_iae=cumulative_iae,
        iae_cm_s=iae,
        sse_cm2_s=sse,
        peak_errors_cm=peak_errors,
        settling_times_s=settling_times,
    )


def _process_diagram(
    result: SimulationResult,
    setpoints: Iterable[float],
    *,
    sensor_max_cm: float = 39.0,
) -> go.Figure:
    """Draw a SCADA P&ID process-flow diagram of the three flotation cells and pumps."""

    setpoint_array = _as_three(setpoints)
    tank_centers = (2.0, 5.7, 9.4)
    tank_width = 2.1
    tank_bottom = 0.4
    tank_top = 4.2
    liquid_height = tank_top - tank_bottom
    final_levels = result.levels_cm[-1]
    final_flows = result.flows_lpm[-1]
    final_inlets = result.disturbances_lpm[-1]

    fig = go.Figure()

    # 1. Main Feed Pipe (Alimentacion externa a Celda 1)
    fig.add_shape(
        type="line", x0=0.1, y0=3.4, x1=tank_centers[0] - tank_width / 2, y1=3.4,
        line=dict(color="#38BDF8", width=8),
    )
    fig.add_annotation(
        x=tank_centers[0] - tank_width / 2 - 0.05, y=3.4, ax=0.2, ay=3.4,
        xref="x", yref="y", axref="x", ayref="y", showarrow=True,
        arrowhead=3, arrowsize=1.4, arrowwidth=2.5, arrowcolor="#0284C7",
    )
    # Feed tag pill
    fig.add_annotation(
        x=0.85, y=3.85,
        text=f"<b>ALIMENTACIÓN Qd</b><br>FIT-00: {final_inlets[0]:.1f} L/min",
        showarrow=False,
        font=dict(size=11, color="#0284C7"),
        bgcolor="rgba(240, 249, 255, 0.9)",
        bordercolor="#38BDF8",
        borderwidth=1,
        borderpad=4,
    )

    # 2. Interconnecting piping between cells
    for pipe_index, (left_center, right_center, flow) in enumerate(
        zip(tank_centers[:-1], tank_centers[1:], final_flows[:-1]), start=1
    ):
        y_pipe = -0.55
        x_pump = left_center
        x_next_inlet = right_center - tank_width / 2

        # Pipe from pump out to next tank inlet
        fig.add_shape(
            type="path",
            path=f"M {x_pump + 0.45},{y_pipe} L {x_next_inlet - 0.25},{y_pipe} L {x_next_inlet - 0.25},3.4 L {x_next_inlet},3.4",
            line=dict(color="#94A3B8", width=6),
        )
        fig.add_annotation(
            x=x_next_inlet, y=3.4, ax=x_next_inlet - 0.35, ay=3.4,
            xref="x", yref="y", axref="x", ayref="y", showarrow=True,
            arrowhead=3, arrowsize=1.3, arrowwidth=2, arrowcolor="#F59E0B",
        )
        fig.add_annotation(
            x=(x_pump + right_center) / 2 - 0.2, y=y_pipe - 0.35,
            text=f"<b>Q{pipe_index} = {flow:.1f} L/min</b>",
            showarrow=False,
            font=dict(size=10, color="#D97706"),
            bgcolor="rgba(254, 243, 199, 0.85)",
            bordercolor="#F59E0B",
            borderwidth=1,
            borderpad=3,
        )

    # 3. Final Tailings outlet
    final_x = tank_centers[-1]
    fig.add_shape(
        type="path",
        path=f"M {final_x + 0.45},-0.55 L 11.4,-0.55",
        line=dict(color="#8B5CF6", width=6),
    )
    fig.add_annotation(
        x=11.45, y=-0.55, ax=10.6, ay=-0.55,
        xref="x", yref="y", axref="x", ayref="y", showarrow=True,
        arrowhead=3, arrowsize=1.4, arrowwidth=2.5, arrowcolor="#7C3AED",
    )
    fig.add_annotation(
        x=11.0, y=-0.15,
        text=f"<b>COLAS FINALES</b><br>Q3 = {final_flows[2]:.1f} L/min",
        showarrow=False,
        font=dict(size=10, color="#6D28D9"),
        bgcolor="rgba(245, 243, 255, 0.9)",
        bordercolor="#8B5CF6",
        borderwidth=1,
        borderpad=4,
    )

    # 4. Tanks, Liquid, Agitators, Froth, Sensors and Pumps
    for i, (center, name) in enumerate(zip(tank_centers, CELL_NAMES)):
        level_fraction = float(np.clip(final_levels[i] / sensor_max_cm, 0.0, 1.0))
        level_top = tank_bottom + liquid_height * level_fraction
        sp_fraction = float(np.clip(setpoint_array[i] / sensor_max_cm, 0.0, 1.0))
        sp_top = tank_bottom + liquid_height * sp_fraction
        left = center - tank_width / 2
        right = center + tank_width / 2

        # Outer Tank Shell with metallic hopper bottom
        fig.add_shape(
            type="rect", x0=left, x1=right, y0=tank_bottom, y1=tank_top,
            line=dict(color="#475569", width=2.5),
            fillcolor="rgba(248, 250, 252, 0.75)",
        )
        # Hopper cone at bottom
        fig.add_shape(
            type="path",
            path=f"M {left},{tank_bottom} L {center - 0.35},{tank_bottom - 0.35} L {center + 0.35},{tank_bottom - 0.35} L {right},{tank_bottom} Z",
            line=dict(color="#475569", width=2),
            fillcolor="rgba(226, 232, 240, 0.85)",
        )

        # Pulp / Slurry Body
        liquid_bottom = tank_bottom - 0.3
        fig.add_shape(
            type="rect", x0=left + 0.05, x1=right - 0.05, y0=tank_bottom,
            y1=max(tank_bottom + 0.05, level_top),
            line=dict(width=0), fillcolor=CELL_COLORS[i], opacity=0.72,
        )

        # Froth / Foam Layer (Mineralized bubbles on top)
        froth_h = min(0.35, level_top - tank_bottom)
        if froth_h > 0.05:
            fig.add_shape(
                type="rect",
                x0=left + 0.05, x1=right - 0.05,
                y0=max(tank_bottom, level_top - froth_h),
                y1=level_top,
                line=dict(color="#F59E0B", width=1.5, dash="dot"),
                fillcolor="rgba(253, 230, 138, 0.8)",
            )
            fig.add_annotation(
                x=right - 0.12, y=level_top - froth_h / 2,
                text="Espuma",
                showarrow=False,
                font=dict(size=8, color="#B45309"),
            )

        # Concentrate overflow lip & launder (Canaleta de concentrado a la derecha)
        fig.add_shape(
            type="path",
            path=f"M {right},{tank_top - 0.25} L {right + 0.3},{tank_top - 0.1} L {right + 0.3},{tank_top - 0.55} L {right},{tank_top - 0.55} Z",
            line=dict(color="#D97706", width=1.5),
            fillcolor="rgba(251, 191, 36, 0.85)",
        )
        fig.add_annotation(
            x=right + 0.18, y=tank_top - 0.05,
            text="Rebose",
            showarrow=False,
            font=dict(size=8, color="#92400E"),
        )

        # Graduation Marks (Ruler on left tank wall)
        for mark_cm in (0, 10, 20, 30, 39):
            mark_y = tank_bottom + liquid_height * (mark_cm / 39.0)
            fig.add_shape(
                type="line", x0=left, y0=mark_y, x1=left + 0.18, y1=mark_y,
                line=dict(color="#64748B", width=1.5),
            )
            fig.add_annotation(
                x=left - 0.08, y=mark_y,
                text=f"{mark_cm}",
                showarrow=False,
                font=dict(size=8, color="#64748B"),
                xanchor="right",
            )

        # Target Setpoint Line (Dashed red/gold target line)
        fig.add_shape(
            type="line", x0=left, y0=sp_top, x1=right, y1=sp_top,
            line=dict(color="#DC2626", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=left + 0.1, y=sp_top + 0.16,
            text=f"SP {setpoint_array[i]:.1f}",
            showarrow=False,
            font=dict(size=9, color="#DC2626"),
            xanchor="left",
            bgcolor="rgba(254, 226, 226, 0.8)",
            borderpad=1,
        )

        # Agitator Motor and Drive Shaft
        fig.add_shape(
            type="rect", x0=center - 0.35, x1=center + 0.35, y0=tank_top, y1=tank_top + 0.45,
            line=dict(color="#334155", width=2),
            fillcolor="#64748B",
        )
        fig.add_annotation(
            x=center, y=tank_top + 0.22,
            text="MOTOR",
            showarrow=False,
            font=dict(size=8, color="#FFFFFF"),
        )
        # Stainless shaft down into cell
        fig.add_shape(
            type="line", x0=center, y0=tank_top, x1=center, y1=tank_bottom + 0.4,
            line=dict(color="#475569", width=4),
        )
        # Impeller blades at bottom
        fig.add_shape(
            type="path",
            path=f"M {center - 0.45},{tank_bottom + 0.55} L {center},{tank_bottom + 0.4} L {center + 0.45},{tank_bottom + 0.55}",
            line=dict(color="#1E293B", width=3),
        )

        # Ultrasonic / Radar Level Sensor (LIT) on top
        fig.add_shape(
            type="rect", x0=center + 0.45, x1=center + 0.85, y0=tank_top, y1=tank_top + 0.3,
            line=dict(color="#0284C7", width=1.5),
            fillcolor="#BAE6FD",
        )
        # Radar beam
        fig.add_shape(
            type="path",
            path=f"M {center + 0.65},{tank_top} L {center + 0.4},{level_top} L {center + 0.9},{level_top} Z",
            line=dict(color="#38BDF8", width=1, dash="dot"),
            fillcolor="rgba(56, 189, 248, 0.12)",
        )
        fig.add_annotation(
            x=center + 0.65, y=tank_top + 0.15,
            text=f"LIT-0{i+1}",
            showarrow=False,
            font=dict(size=7, color="#0369A1"),
        )

        # Tank Title Banner
        fig.add_annotation(
            x=center, y=tank_top + 0.72,
            text=f"<b>{name.upper()}</b>",
            showarrow=False,
            font=dict(size=14, color="#0F172A"),
        )

        # Current Level Badge & Error Pill
        err = final_levels[i] - setpoint_array[i]
        err_color = "#16A34A" if abs(err) <= 0.25 else ("#D97706" if abs(err) <= 1.0 else "#DC2626")
        fig.add_annotation(
            x=center, y=level_top + 0.28,
            text=f"<b>{final_levels[i]:.2f} cm</b> (Δ {err:+.2f})",
            showarrow=False,
            font=dict(size=11, color=err_color),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor=err_color,
            borderwidth=1.5,
            borderpad=3,
        )

        # Peristaltic Pump underneath
        pump_cx = center
        pump_cy = -0.55
        pump_r = 0.45
        # Pipe from hopper to pump
        fig.add_shape(
            type="line", x0=center, y0=tank_bottom - 0.35, x1=center, y1=pump_cy + pump_r,
            line=dict(color="#64748B", width=4),
        )
        # Pump outer casing
        fig.add_shape(
            type="circle",
            x0=pump_cx - pump_r, x1=pump_cx + pump_r,
            y0=pump_cy - pump_r, y1=pump_cy + pump_r,
            line=dict(color="#334155", width=2.5),
            fillcolor="rgba(241, 245, 249, 0.95)",
        )
        # Pump rotor rollers (3 peristaltic rollers)
        for angle_deg in (0, 120, 240):
            rad = np.radians(angle_deg)
            rx = pump_cx + 0.22 * np.cos(rad)
            ry = pump_cy + 0.22 * np.sin(rad)
            fig.add_shape(
                type="circle",
                x0=rx - 0.08, x1=rx + 0.08, y0=ry - 0.08, y1=ry + 0.08,
                line=dict(color="#0284C7", width=1.5),
                fillcolor="#38BDF8",
            )
        fig.add_annotation(
            x=pump_cx, y=pump_cy - 0.72,
            text=f"<b>Bomba P{i+1}</b><br>{final_flows[i]:.1f} L/min",
            showarrow=False,
            font=dict(size=10, color="#1E293B"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=35, b=20),
        xaxis=dict(range=[-0.2, 12.0], visible=False, fixedrange=True),
        yaxis=dict(range=[-1.55, 5.25], visible=False, fixedrange=True),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _plot_level_trends(
    with_ff: SimulationResult,
    feedback_only: SimulationResult,
    setpoints: Iterable[float],
    compare: bool,
    disturbance_start_s: float,
) -> go.Figure:
    """Multi-cell level dynamic response with tolerance band and disturbance event."""

    setpoint_array = _as_three(setpoints)
    fig = go.Figure()

    for index, name in enumerate(CELL_NAMES):
        sp = setpoint_array[index]
        color = CELL_COLORS[index]

        # Shaded tolerance envelope (+/- 0.5 cm)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([with_ff.time_s, with_ff.time_s[::-1]]),
                y=np.concatenate([np.full_like(with_ff.time_s, sp + 0.5), np.full_like(with_ff.time_s, sp - 0.5)]),
                fill="toself",
                fillcolor=CELL_BG_COLORS[index],
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"Tolerancia ±0.5cm {name}",
            )
        )

        # Setpoint line
        fig.add_trace(
            go.Scatter(
                x=with_ff.time_s,
                y=np.full_like(with_ff.time_s, sp),
                mode="lines",
                name=f"Setpoint {name} ({sp:.1f} cm)",
                line=dict(color=color, width=1.2, dash="dot"),
                hovertemplate=f"Setpoint {name}: {sp:.1f} cm<extra></extra>",
            )
        )

        # PI + FF curve
        fig.add_trace(
            go.Scatter(
                x=with_ff.time_s,
                y=with_ff.levels_cm[:, index],
                mode="lines",
                name=f"{name} · PI + Feedforward",
                line=dict(color=color, width=2.8),
                hovertemplate=f"{name} (PI+FF): %{{y:.2f}} cm<extra></extra>",
            )
        )

        # PI only curve (if comparing)
        if compare:
            fig.add_trace(
                go.Scatter(
                    x=feedback_only.time_s,
                    y=feedback_only.levels_cm[:, index],
                    mode="lines",
                    name=f"{name} · PI Solamente",
                    line=dict(color=color, width=1.8, dash="dash"),
                    hovertemplate=f"{name} (PI Solo): %{{y:.2f}} cm<extra></extra>",
                )
            )

    # Disturbance marker
    fig.add_vline(
        x=disturbance_start_s,
        line_width=2,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="⚡ Perturbación en Qd",
        annotation_position="top left",
        annotation_font=dict(color="#DC2626", size=11),
    )

    fig.update_layout(
        title="<b>Dinámica Temporal de Niveles de Pulpa vs Setpoint</b>",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Nivel de Pulpa (cm)",
        yaxis=dict(range=[0, 39], gridcolor="rgba(148, 163, 184, 0.2)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.2)"),
        hovermode="x unified",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
    )
    return fig


def _plot_control_efforts(with_ff: SimulationResult, feedback_only: SimulationResult, compare: bool) -> go.Figure:
    """Plot peristaltic pump flow actions with actuator saturation limits."""

    fig = go.Figure()

    # Upper and lower pump limits
    fig.add_hline(
        y=100.0, line_width=1.5, line_dash="dot", line_color="#EF4444",
        annotation_text="Qmax = 100 L/min", annotation_position="bottom right",
    )
    fig.add_hline(
        y=0.0, line_width=1.5, line_dash="dot", line_color="#EF4444",
        annotation_text="Qmin = 0 L/min", annotation_position="top right",
    )

    for i, name in enumerate(CELL_NAMES):
        fig.add_trace(
            go.Scatter(
                x=with_ff.time_s,
                y=with_ff.flows_lpm[:, i],
                mode="lines",
                name=f"Bomba P{i+1} · PI+FF",
                line=dict(color=CELL_COLORS[i], width=2.4),
                hovertemplate=f"Q{i+1} (PI+FF): %{{y:.2f}} L/min<extra></extra>",
            )
        )
        if compare:
            fig.add_trace(
                go.Scatter(
                    x=feedback_only.time_s,
                    y=feedback_only.flows_lpm[:, i],
                    mode="lines",
                    name=f"Bomba P{i+1} · PI Solo",
                    line=dict(color=CELL_COLORS[i], width=1.5, dash="dash"),
                    hovertemplate=f"Q{i+1} (PI Solo): %{{y:.2f}} L/min<extra></extra>",
                )
            )

    fig.update_layout(
        title="<b>Acciones de Control (Caudales de Salida de Cola)</b>",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Caudal de Bomba (L/min)",
        yaxis=dict(range=[-2, 105], gridcolor="rgba(148, 163, 184, 0.2)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.2)"),
        hovermode="x unified",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
    )
    return fig


def _plot_cumulative_iae(with_ff: SimulationResult, feedback_only: SimulationResult, compare: bool) -> go.Figure:
    """Plot cumulative IAE integral over time showing dynamic error accumulation."""

    fig = go.Figure()
    for i, name in enumerate(CELL_NAMES):
        fig.add_trace(
            go.Scatter(
                x=with_ff.time_s,
                y=with_ff.cumulative_iae[:, i],
                mode="lines",
                name=f"IAE {name} · PI+FF",
                line=dict(color=CELL_COLORS[i], width=2.5),
                hovertemplate=f"IAE {name} (PI+FF): %{{y:.2f}} cm·s<extra></extra>",
            )
        )
        if compare:
            fig.add_trace(
                go.Scatter(
                    x=feedback_only.time_s,
                    y=feedback_only.cumulative_iae[:, i],
                    mode="lines",
                    name=f"IAE {name} · PI Solo",
                    line=dict(color=CELL_COLORS[i], width=1.6, dash="dash"),
                    hovertemplate=f"IAE {name} (PI Solo): %{{y:.2f}} cm·s<extra></extra>",
                )
            )

    fig.update_layout(
        title="<b>Evolución del IAE Acumulado: ∫ |e(t)| dt</b>",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="IAE Acumulado (cm·s)",
        grid_pattern="independent",
        hovermode="x unified",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
    )
    return fig


def _metrics_table(result: SimulationResult, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Estrategia": [label] * 3,
            "Celda": CELL_NAMES,
            "IAE (cm·s)": np.round(result.iae_cm_s, 3),
            "SSE (cm²·s)": np.round(result.sse_cm2_s, 3),
            "Desviación Máx (cm)": np.round(result.peak_errors_cm, 3),
            "Error Final (cm)": np.round(result.errors_cm[-1], 4),
            "Tiempo Asent. (s)": [
                f"{t:.1f}" if not np.isnan(t) else "N/A" for t in result.settling_times_s
            ],
            "Caudal Medio (L/min)": np.round(result.flows_lpm.mean(axis=0), 2),
        }
    )


def _results_dataframe(result: SimulationResult, setpoints: Iterable[float]) -> pd.DataFrame:
    setpoint_array = _as_three(setpoints)
    data: Dict[str, np.ndarray] = {"Tiempo (s)": result.time_s}
    for i, name in enumerate(CELL_NAMES):
        data[f"{name} - Setpoint (cm)"] = np.full(result.time_s.shape, setpoint_array[i])
        data[f"{name} - Nivel (cm)"] = result.levels_cm[:, i]
        data[f"{name} - Caudal Bomba (L/min)"] = result.flows_lpm[:, i]
        data[f"{name} - Perturbacion Entrada (L/min)"] = result.disturbances_lpm[:, i]
        data[f"{name} - Error (cm)"] = result.errors_cm[:, i]
        data[f"{name} - IAE Acumulado (cm·s)"] = result.cumulative_iae[:, i]
    return pd.DataFrame(data).round(5)


def _article_parameters_table(parameters: ArticleParameters = ARTICLE) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Celda": CELL_NAMES,
            "Ganancia Integrador k' [(cm·s⁻¹)/(L/min)]": parameters.k_integrating,
            "Retardo de Transporte θ (s)": parameters.delay_s,
            "Ganancia Proporcional Kc [(L/min)/cm]": parameters.kc,
            "Tiempo Integral Ti (s)": parameters.ti_s,
            "Compensador Feedforward Kff": parameters.kff,
        }
    ).round(6)


def main() -> None:
    # Custom CSS for Industrial SCADA UI
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1480px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .scada-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 14px;
            padding: 1.25rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }
        .scada-title {
            color: #F8FAFC;
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .scada-subtitle {
            color: #94A3B8;
            font-size: 0.95rem;
            margin-top: 0.35rem;
            margin-bottom: 0.5rem;
        }
        .badge-pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 9999px;
            margin-right: 0.4rem;
        }
        .badge-cyan { background: rgba(6, 182, 212, 0.2); color: #38BDF8; border: 1px solid #0284C7; }
        .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ADE80; border: 1px solid #16A34A; }
        .badge-purple { background: rgba(139, 92, 246, 0.2); color: #C084FC; border: 1px solid #7C3AED; }
        
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border-left: 4px solid #0EA5E9;
        }
        .kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 600;
            color: #64748B;
            letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 0.2rem;
        }
        .kpi-delta {
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }
        .delta-positive { color: #16A34A; }
        .delta-negative { color: #DC2626; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # SCADA Top Banner
    st.markdown(
        """
        <div class="scada-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 class="scada-title">⚗️ Control de Nivel en Batería de Flotación · Gemelo Digital SCADA</h1>
                    <div class="scada-subtitle">
                        Simulación dinámica basada en <b>Quintanilla et al. (MethodsX 2023)</b> — 3 celdas de flotación en serie con bombas peristálticas, retardos de transporte y desacoplo Feedforward.
                    </div>
                </div>
            </div>
            <div style="margin-top: 0.4rem;">
                <span class="badge-pill badge-green">● SISTEMA EN LÍNEA</span>
                <span class="badge-pill badge-cyan">CONTROL: FEEDBACK PI + FEEDFORWARD</span>
                <span class="badge-pill badge-purple">PLANTA: INTEGRADORA CON RETARDO</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Preset Scenarios & Simulation Controls
    with st.sidebar:
        st.markdown("### 🎛️ Escenarios y Presets")
        preset_choice = st.selectbox(
            "Cargar Escenario Rápido:",
            options=[
                "🧪 Caso Base (Quintanilla et al. 2023)",
                "⚡ Perturbación Severa (+25 L/min)",
                "🔻 Reducción de Caudal (-15 L/min)",
                "🌊 Cascada de Setpoints Desfasados",
            ],
            index=0,
            help="Selecciona un escenario preconfigurado para probar el controlador bajo distintas condiciones.",
        )

        # Default values based on preset
        if "Caso Base" in preset_choice:
            def_dur, def_dt, def_qnom, def_tpert, def_dq = 900.0, 1.0, 45.0, 300.0, 10.0
            def_sp1, def_sp2, def_sp3 = 20.0, 20.0, 20.0
        elif "Severa" in preset_choice:
            def_dur, def_dt, def_qnom, def_tpert, def_dq = 900.0, 1.0, 45.0, 200.0, 25.0
            def_sp1, def_sp2, def_sp3 = 20.0, 20.0, 20.0
        elif "Reducción" in preset_choice:
            def_dur, def_dt, def_qnom, def_tpert, def_dq = 900.0, 1.0, 45.0, 250.0, -15.0
            def_sp1, def_sp2, def_sp3 = 20.0, 20.0, 20.0
        else:  # Cascada de Setpoints
            def_dur, def_dt, def_qnom, def_tpert, def_dq = 900.0, 1.0, 45.0, 300.0, 10.0
            def_sp1, def_sp2, def_sp3 = 24.0, 20.0, 16.0

        st.divider()
        st.markdown("### ⏱️ Parámetros de Operación")
        duration = st.number_input(
            "Duración Total (s)", min_value=60.0, max_value=3600.0, value=def_dur, step=60.0,
            help="Tiempo total de la prueba virtual en segundos.",
        )
        dt = st.select_slider(
            "Paso de Integración Δt (s)", options=[0.25, 0.5, 1.0, 2.0, 5.0], value=def_dt,
            help="Paso temporal de integración numérica.",
        )
        nominal_flow = st.number_input(
            "Caudal Nominal Qd,nom (L/min)", min_value=5.0, max_value=80.0, value=def_qnom, step=1.0,
            help="Punto de equilibrio inicial de la batería de celdas.",
        )
        disturbance_start = st.number_input(
            "Instante del Escalón (s)", min_value=0.0, max_value=float(duration),
            value=min(def_tpert, float(duration) * 0.8), step=10.0,
            help="Tiempo en el que se inyecta la variación de caudal de alimentación.",
        )
        disturbance_step = st.number_input(
            "Salto de Caudal ΔQd (L/min)", min_value=-40.0, max_value=40.0, value=def_dq, step=1.0,
            help="Magnitud de la perturbación escalón en la celda 1.",
        )

        st.divider()
        st.markdown("### 🎯 Consignas de Nivel (Setpoints)")
        col_sp1, col_sp2, col_sp3 = st.columns(3)
        with col_sp1:
            sp1 = st.number_input("SP Celda 1", min_value=2.0, max_value=37.0, value=def_sp1, step=1.0)
        with col_sp2:
            sp2 = st.number_input("SP Celda 2", min_value=2.0, max_value=37.0, value=def_sp2, step=1.0)
        with col_sp3:
            sp3 = st.number_input("SP Celda 3", min_value=2.0, max_value=37.0, value=def_sp3, step=1.0)

        compare = st.checkbox("Comparar Feedforward vs Solo Feedback", value=True)

        # Advanced Tuning Expander
        with st.expander("🛠️ Sintonización Avanzada de Controladores"):
            custom_tuning = st.checkbox("Habilitar Sintonización Manual", value=False)
            if custom_tuning:
                st.caption("Modifica las ganancias respecto al artículo:")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    kc1 = st.number_input("Kc Celda 1", value=-62.0, step=5.0)
                    ti1 = st.number_input("Ti Celda 1 (s)", value=56.0, step=5.0)
                    kff1 = st.number_input("Kff Celda 1", value=0.8, step=0.1)
                with col_t2:
                    kc2 = st.number_input("Kc Celda 2", value=-76.0, step=5.0)
                    ti2 = st.number_input("Ti Celda 2 (s)", value=30.0, step=5.0)
                    kff2 = st.number_input("Kff Celda 2", value=0.8, step=0.1)
                kc3 = st.number_input("Kc Celda 3", value=-85.0, step=5.0)
                ti3 = st.number_input("Ti Celda 3 (s)", value=44.0, step=5.0)
                kff3 = st.number_input("Kff Celda 3", value=1.0, step=0.1)
                active_params = ArticleParameters(
                    k_integrating=ARTICLE.k_integrating,
                    delay_s=ARTICLE.delay_s,
                    kc=np.array([kc1, kc2, kc3], dtype=float),
                    ti_s=np.array([ti1, ti2, ti3], dtype=float),
                    kff=np.array([kff1, kff2, kff3], dtype=float),
                )
            else:
                active_params = ARTICLE

        st.divider()
        st.caption("Rango físico del sensor ultrasónico: 0 a 39 cm")

    # Run Simulations
    setpoints = (sp1, sp2, sp3)
    with_ff = simulate(
        duration_s=float(duration),
        dt_s=float(dt),
        setpoints_cm=setpoints,
        initial_levels_cm=setpoints,
        nominal_flow_lpm=float(nominal_flow),
        disturbance_start_s=float(disturbance_start),
        disturbance_step_lpm=float(disturbance_step),
        use_feedforward=True,
        parameters=active_params,
    )
    feedback_only = simulate(
        duration_s=float(duration),
        dt_s=float(dt),
        setpoints_cm=setpoints,
        initial_levels_cm=setpoints,
        nominal_flow_lpm=float(nominal_flow),
        disturbance_start_s=float(disturbance_start),
        disturbance_step_lpm=float(disturbance_step),
        use_feedforward=False,
        parameters=active_params,
    )

    # Performance KPI Calculations
    total_iae_ff = float(np.sum(with_ff.iae_cm_s))
    total_iae_fb = float(np.sum(feedback_only.iae_cm_s))
    reduction_pct = 100.0 * (1.0 - total_iae_ff / max(total_iae_fb, 1e-12))
    max_peak_error_ff = float(np.max(with_ff.peak_errors_cm))
    max_peak_error_fb = float(np.max(feedback_only.peak_errors_cm))
    final_max_error = float(np.max(np.abs(with_ff.errors_cm[-1])))

    # Executive KPI Dashboard Cards
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #0284C7;">
                <div class="kpi-label">IAE Total · PI + Feedforward</div>
                <div class="kpi-value">{total_iae_ff:.2f} <span style="font-size: 1rem; color: #64748B;">cm·s</span></div>
                <div class="kpi-delta delta-positive">PI Solo: {total_iae_fb:.2f} cm·s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #16A34A;">
                <div class="kpi-label">Reducción de Error IAE</div>
                <div class="kpi-value" style="color: #16A34A;">{reduction_pct:+.1f}%</div>
                <div class="kpi-delta delta-positive">Mejora por desacoplo Feedforward</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #F59E0B;">
                <div class="kpi-label">Desviación Pico Máxima</div>
                <div class="kpi-value">{max_peak_error_ff:.2f} <span style="font-size: 1rem; color: #64748B;">cm</span></div>
                <div class="kpi-delta">Sin FF: {max_peak_error_fb:.2f} cm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[3]:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #8B5CF6;">
                <div class="kpi-label">Error de Régimen Permanente</div>
                <div class="kpi-value">{final_max_error:.4f} <span style="font-size: 1rem; color: #64748B;">cm</span></div>
                <div class="kpi-delta delta-positive">Estabilidad: 100% Asentado</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Main Tabs
    tab_scada, tab_analytics, tab_actuators, tab_theory = st.tabs(
        [
            "🖥️ Monitor SCADA & Curvas",
            "📊 Análisis Comparativo (PI vs PI+FF)",
            "🎛️ Diagnóstico de Bombas y Actuadores",
            "📚 Modelo Matemático & Metodología",
        ]
    )

    with tab_scada:
        st.markdown("#### 🏭 Diagrama de Proceso P&ID Interactivo")
        st.caption("Visualización en tiempo real del banco de 3 celdas, capa de espuma, mecanismo de agitación, bombas peristálticas y telemetría de caudal.")
        st.plotly_chart(
            _process_diagram(with_ff, setpoints),
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        )

        st.markdown("#### 📈 Respuesta Dinámica de Nivel")
        st.caption("Línea continua: PI + Feedforward | Línea discontinua: PI Feedback solamente | Área sombreada: Banda de tolerancia ±0.5 cm")
        st.plotly_chart(
            _plot_level_trends(with_ff, feedback_only, setpoints, compare, disturbance_start),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with tab_analytics:
        st.markdown("#### ⚖️ Comparativa de Rendimiento de Control")
        col_iaec, col_table = st.columns([1, 1])
        with col_iaec:
            st.plotly_chart(
                _plot_cumulative_iae(with_ff, feedback_only, compare),
                use_container_width=True,
                config={"displaylogo": False},
            )
        with col_table:
            st.markdown("##### Métricas Cuantitativas por Celda")
            metric_tables = [_metrics_table(with_ff, "PI + Feedforward")]
            if compare:
                metric_tables.append(_metrics_table(feedback_only, "PI Solamente"))
            metrics_df = pd.concat(metric_tables, ignore_index=True)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            improvement_per_cell = 100.0 * (1.0 - with_ff.iae_cm_s / np.maximum(feedback_only.iae_cm_s, 1e-12))
            st.success(
                f"✅ **Efectividad del Feedforward:** Reducción de IAE de **{improvement_per_cell[0]:.1f}%** en Celda 1, "
                f"**{improvement_per_cell[1]:.1f}%** en Celda 2 y **{improvement_per_cell[2]:.1f}%** en Celda 3."
            )

        st.divider()
        st.markdown("#### 💾 Exportación de Datos de Simulación")
        csv_bytes = _results_dataframe(with_ff, setpoints).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar Serie Temporal Completa (CSV)",
            data=csv_bytes,
            file_name="simulacion_control_flotacion_scada.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption("El dataset incluye tiempo, setpoints, niveles, caudales de bombas, perturbaciones e IAE acumulado.")

    with tab_actuators:
        st.markdown("#### 🎛️ Diagnóstico de Bombas Peristálticas y Caudales")
        st.caption("Análisis de esfuerzo de control de las bombas P1, P2 y P3 ante la propagación del escalón de caudal.")
        st.plotly_chart(
            _plot_control_efforts(with_ff, feedback_only, compare),
            use_container_width=True,
            config={"displaylogo": False},
        )

        col_flow1, col_flow2 = st.columns(2)
        with col_flow1:
            # Flow distribution chart
            fig_dist = go.Figure()
            for i, name in enumerate(CELL_NAMES):
                fig_dist.add_trace(
                    go.Scatter(
                        x=with_ff.time_s,
                        y=with_ff.disturbances_lpm[:, i],
                        mode="lines",
                        name=f"Perturbación {name}",
                        line=dict(color=CELL_COLORS[i], width=2),
                    )
                )
            fig_dist.update_layout(
                title="<b>Perturbaciones de Entrada a Cada Celda</b>",
                xaxis_title="Tiempo (s)",
                yaxis_title="Caudal de Entrada (L/min)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_dist, use_container_width=True, config={"displaylogo": False})

        with col_flow2:
            # Errors distribution
            fig_err = go.Figure()
            for i, name in enumerate(CELL_NAMES):
                fig_err.add_trace(
                    go.Scatter(
                        x=with_ff.time_s,
                        y=with_ff.errors_cm[:, i],
                        mode="lines",
                        name=f"Error {name}",
                        line=dict(color=CELL_COLORS[i], width=2),
                    )
                )
            fig_err.update_layout(
                title="<b>Desviación de Nivel e(t) = SP - h(t)</b>",
                xaxis_title="Tiempo (s)",
                yaxis_title="Error (cm)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_err, use_container_width=True, config={"displaylogo": False})

    with tab_theory:
        st.markdown("#### 📚 Formulación Matemática y Metodología")
        st.markdown(
            r"""
            La simulación reproduce el modelo dinámico y la estrategia de control implementada experimentalmente en:

            > **P. Quintanilla, D. Navia, F. Moreno et al.**, *“A methodology to implement a closed-loop feedback-feedforward level control in a laboratory-scale flotation bank using peristaltic pumps”*, **MethodsX** 10 (2023) 102081. DOI: [10.1016/j.mex.2023.102081](https://doi.org/10.1016/j.mex.2023.102081).
            """
        )

        st.markdown("##### 1. Modelo de Planta Integradora con Retardo")
        st.latex(r"G_{p,i}(s) = \frac{k'_i e^{-\theta_i s}}{s}")
        st.markdown(
            r"""
            Donde:
            - $k'_i$: ganancia del integrador $[(\text{cm}\cdot\text{s}^{-1})/(\text{L/min})]$, derivada de la geometría de la celda de flotación.
            - $\theta_i$: retardo puro de transporte $(\text{s})$ asociado a la manguera y dinámica de la bomba peristáltica.
            - $s$: operador de Laplace (sistema integrador puro tipo 1).
            """
        )

        st.markdown("##### 2. Ley de Control Feedback PI con Compensación Feedforward")
        st.latex(
            r"u_i(t) = \bar{u}_i + K_{c,i}\left(e_i(t) + \frac{1}{\tau_{I,i}} \int_0^t e_i(\tau)d\tau\right) + K_{ff,i}\left(Q_{d,i}(t) - \bar{Q}_d\right)"
        )
        st.markdown(
            r"""
            Donde:
            - $\bar{u}_i = Q_{\text{nominal}}$: sesgo de operación para mantener estacionario el estado inicial.
            - $e_i(t) = SP_i - h_i(t)$: error instantáneo de nivel de pulpa.
            - $K_{c,i}$: ganancia proporcional $[(\text{L/min})/\text{cm}]$.
            - $\tau_{I,i}$: constante de tiempo integral $(\text{s})$.
            - $K_{ff,i}$: ganancia del compensador anticipativo feedforward frente a perturbaciones de caudal de alimentación.
            """
        )

        st.markdown("##### 3. Parámetros Experimentales Identificados (Tabla 2 del Artículo)")
        st.dataframe(_article_parameters_table(active_params), use_container_width=True, hide_index=True)

        with st.expander("ℹ️ Alcance y Consideraciones"):
            st.markdown(
                """
                - **Desacoplo:** El banco de flotación conecta las celdas en serie: el caudal de colas de la celda 1 ingresa a la celda 2, y el de la celda 2 a la celda 3. La compensación feedforward mitiga la propagación del transitorio a través de la batería.
                - **Límites Físicos:** El actuador tiene una carrera de saturación de 0 a 100 L/min, y el sensor de nivel opera en el rango físico de 0 a 39 cm.
                - **Diferencia con el montaje físico:** La simulación se enfoca en la respuesta de control de nivel hidrodinámico; no incluye variables químicas (colectores, espumantes), aireación forzada ni recuperación metalúrgica de concentrado.
                """
            )

    st.markdown(
        """
        <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; margin-top: 2rem; padding: 1rem; border-top: 1px solid #E2E8F0;">
            🔬 Gemelo Digital de Flotación · Basado en Quintanilla et al. (MethodsX 2023) · Desplegable en Streamlit / Render / Vercel
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
