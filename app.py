"""Interactive simulation of the three-cell flotation level-control case.

The implementation follows the mathematical relationships reported in:
Quintanilla et al., MethodsX 10 (2023) 102081, doi:10.1016/j.mex.2023.102081.

The plant is represented as an integrating process with transport delay,
G_i(s) = k'_i exp(-theta_i s) / s.  The PI controller is implemented in
incremental form around a nominal flow so that the initial state is at a
steady operating point.  This is equivalent to the article's PI law when the
nominal operating point is included as the controller bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Control de nivel en flotación",
    page_icon="⚗️",
    layout="wide",
)


CELL_NAMES = ("Celda 1", "Celda 2", "Celda 3")
CELL_COLORS = ("#00A6A6", "#F59E0B", "#7C3AED")


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
    iae_cm_s: np.ndarray
    sse_cm2_s: np.ndarray


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

    The disturbance of cell 1 is the external feed flow.  The disturbances of
    cells 2 and 3 are the previous cell's tailings flow, as in Fig. 15 of the
    article.  The manipulated variables are the three tailings flows.
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

    # The nominal tailings flow is the bias of each PI loop.  It makes the
    # initial condition stationary when inlet and outlet flows are equal.
    nominal = float(nominal_flow_lpm)
    previous_disturbance = np.full(3, nominal, dtype=float)
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
        delta_disturbance = disturbances[k] - previous_disturbance
        feedforward_term = parameters.kff * delta_disturbance if use_feedforward else 0.0

        # Eq. (7) from the article plus the nominal-flow bias.
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

        # Integrating model Gp(s) = k' exp(-theta s) / s from Table 2.
        # The disturbance enters with the opposite sign to the tailings flow.
        level_derivative = parameters.k_integrating * (delayed_flows - disturbances[k])
        levels[k + 1] = np.clip(levels[k] + level_derivative * dt_s, 0.0, 39.0)
        previous_disturbance = disturbances[k]

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
    return SimulationResult(
        time_s=time_s,
        levels_cm=levels,
        flows_lpm=flows,
        disturbances_lpm=disturbances,
        errors_cm=errors,
        iae_cm_s=iae,
        sse_cm2_s=sse,
    )


def _line_chart(
    time_s: np.ndarray,
    series: Dict[str, np.ndarray],
    *,
    y_title: str,
    title: str,
    dashed: bool = False,
) -> go.Figure:
    fig = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=time_s,
                y=values,
                mode="lines",
                name=name,
                line=dict(
                    color=CELL_COLORS[index % len(CELL_COLORS)],
                    width=2,
                    dash="dash" if dashed else "solid",
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo (s)",
        yaxis_title=y_title,
        hovermode="x unified",
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _metrics_table(result: SimulationResult, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Estrategia": [label] * 3,
            "Celda": CELL_NAMES,
            "IAE (cm·s)": np.round(result.iae_cm_s, 3),
            "SSE (cm²·s)": np.round(result.sse_cm2_s, 3),
            "Error final (cm)": np.round(result.errors_cm[-1], 3),
            "Caudal medio (L/min)": np.round(result.flows_lpm.mean(axis=0), 3),
        }
    )


def _results_dataframe(result: SimulationResult, setpoints: Iterable[float]) -> pd.DataFrame:
    """Flatten the simulation arrays into a teacher-friendly CSV table."""

    setpoint_array = _as_three(setpoints)
    data: Dict[str, np.ndarray] = {"Tiempo (s)": result.time_s}
    for i, name in enumerate(CELL_NAMES):
        data[f"{name} - Setpoint (cm)"] = np.full(result.time_s.shape, setpoint_array[i])
        data[f"{name} - Nivel (cm)"] = result.levels_cm[:, i]
        data[f"{name} - Caudal de cola (L/min)"] = result.flows_lpm[:, i]
        data[f"{name} - Perturbación (L/min)"] = result.disturbances_lpm[:, i]
        data[f"{name} - Error (cm)"] = result.errors_cm[:, i]
    return pd.DataFrame(data).round(6)


def _article_parameters_table(parameters: ArticleParameters = ARTICLE) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Celda": CELL_NAMES,
            "k' (cm·s⁻¹)/(L/min)": parameters.k_integrating,
            "Retardo θ (s)": parameters.delay_s,
            "Kc (L/min)/cm": parameters.kc,
            "Ti (s)": parameters.ti_s,
            "Kff": parameters.kff,
        }
    ).round(4)


def _process_diagram(
    result: SimulationResult,
    setpoints: Iterable[float],
    *,
    sensor_max_cm: float = 39.0,
) -> go.Figure:
    """Draw a process-flow view of the three flotation tanks and pumps."""

    setpoint_array = _as_three(setpoints)
    tank_centers = (1.7, 4.8, 7.9)
    tank_width = 1.65
    tank_bottom = 0.15
    tank_top = 3.35
    liquid_height = tank_top - tank_bottom
    final_levels = result.levels_cm[-1]
    final_flows = result.flows_lpm[-1]
    final_inlets = result.disturbances_lpm[-1]

    fig = go.Figure()

    # Pipes and arrows are drawn first so the tanks and pumps remain legible.
    fig.add_shape(
        type="line", x0=0.05, y0=2.65, x1=tank_centers[0] - tank_width / 2, y1=2.65,
        line=dict(color="#94A3B8", width=7),
    )
    fig.add_annotation(
        x=tank_centers[0] - tank_width / 2, y=2.65, ax=0.1, ay=2.65,
        xref="x", yref="y", axref="x", ayref="y", showarrow=True,
        arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#00A6A6",
    )
    fig.add_annotation(x=0.35, y=2.95, text="Alimentación Qd", showarrow=False, font=dict(size=12))

    for pipe_index, (left_center, right_center, flow) in enumerate(
        zip(tank_centers[:-1], tank_centers[1:], final_flows[:-1]), start=1
    ):
        y_pipe = 0.95
        x_start = left_center + tank_width / 2
        x_end = right_center - tank_width / 2
        fig.add_shape(
            type="line", x0=x_start, y0=y_pipe, x1=x_end, y1=y_pipe,
            line=dict(color="#94A3B8", width=7),
        )
        fig.add_annotation(
            x=x_end - 0.08, y=y_pipe, ax=x_start + 0.08, ay=y_pipe,
            xref="x", yref="y", axref="x", ayref="y", showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#F59E0B",
        )
        fig.add_annotation(
            x=(x_start + x_end) / 2, y=y_pipe + 0.28,
            text=f"Q{pipe_index} = {flow:.1f} L/min",
            showarrow=False, font=dict(size=11, color="#64748B"),
        )

    # Final-cell tailings outlet.
    final_outlet_start = tank_centers[-1] + tank_width / 2
    fig.add_shape(
        type="line", x0=final_outlet_start, y0=0.95, x1=9.35, y1=0.95,
        line=dict(color="#94A3B8", width=7),
    )
    fig.add_annotation(
        x=9.35, y=0.95, ax=final_outlet_start + 0.05, ay=0.95,
        xref="x", yref="y", axref="x", ayref="y", showarrow=True,
        arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#7C3AED",
    )
    fig.add_annotation(x=8.9, y=1.28, text=f"Salida = {final_flows[2]:.1f} L/min", showarrow=False, font=dict(size=11, color="#64748B"))

    for i, (center, name) in enumerate(zip(tank_centers, CELL_NAMES)):
        level_fraction = float(np.clip(final_levels[i] / sensor_max_cm, 0.0, 1.0))
        level_top = tank_bottom + liquid_height * level_fraction
        left = center - tank_width / 2
        right = center + tank_width / 2

        # Liquid and tank shell.
        fig.add_shape(
            type="rect", x0=left + 0.08, x1=right - 0.08, y0=tank_bottom,
            y1=max(tank_bottom + 0.02, level_top),
            line=dict(width=0), fillcolor=CELL_COLORS[i], opacity=0.55,
        )
        fig.add_shape(
            type="rect", x0=left, x1=right, y0=tank_bottom, y1=tank_top,
            line=dict(color="#64748B", width=2), fillcolor="rgba(148,163,184,0.08)",
        )
        fig.add_shape(
            type="line", x0=left + 0.08, x1=right - 0.08, y0=level_top, y1=level_top,
            line=dict(color=CELL_COLORS[i], width=2),
        )

        # Peristaltic pump under each tank.
        pump_y0, pump_y1 = -0.85, -0.35
        fig.add_shape(
            type="rect", x0=center - 0.52, x1=center + 0.52, y0=pump_y0, y1=pump_y1,
            line=dict(color="#475569", width=1.5), fillcolor="rgba(71,85,105,0.22)",
        )
        fig.add_shape(
            type="circle", x0=center - 0.16, x1=center + 0.16, y0=pump_y0 + 0.09, y1=pump_y1 - 0.09,
            line=dict(color="#00A6A6", width=2), fillcolor="rgba(0,166,166,0.25)",
        )
        fig.add_shape(
            type="line", x0=center, x1=center, y0=tank_bottom, y1=pump_y1,
            line=dict(color="#94A3B8", width=3, dash="dot"),
        )

        fig.add_annotation(x=center, y=3.65, text=f"<b>{name}</b>", showarrow=False, font=dict(size=13))
        fig.add_annotation(x=center, y=level_top + 0.22, text=f"{final_levels[i]:.2f} cm", showarrow=False, font=dict(size=12, color=CELL_COLORS[i]))
        fig.add_annotation(x=center, y=-1.12, text=f"Bomba P{i + 1}<br>{final_flows[i]:.1f} L/min", showarrow=False, font=dict(size=10, color="#64748B"))
        fig.add_annotation(x=center, y=3.12, text=f"Entrada: {final_inlets[i]:.1f} L/min", showarrow=False, font=dict(size=10, color="#64748B"))
        fig.add_annotation(x=right - 0.06, y=0.32, text=f"SP {setpoint_array[i]:.1f}", showarrow=False, xanchor="right", font=dict(size=10, color="#64748B"))

    fig.update_layout(
        height=370,
        margin=dict(l=10, r=10, t=30, b=12),
        xaxis=dict(range=[-0.1, 9.55], visible=False, fixedrange=True),
        yaxis=dict(range=[-1.35, 4.05], visible=False, fixedrange=True),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1450px; padding-top: 2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        .app-kicker {color: #00a6a6; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .78rem;}
        .app-subtitle {color: #718096; font-size: 1.02rem; margin-top: -.45rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='app-kicker'>Laboratorio de control de procesos</div>", unsafe_allow_html=True)
    st.title("Control de nivel en una batería de flotación")
    st.markdown(
        "<div class='app-subtitle'>Simulación interactiva del artículo de Quintanilla et al. "
        "con tres celdas en serie, retardos, PI y feedforward.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Escenario de simulación")
        duration = st.number_input(
            "Duración (s)", min_value=60.0, max_value=3600.0, value=900.0, step=60.0,
            help="Tiempo total del experimento virtual.",
        )
        dt = st.select_slider(
            "Paso de integración (s)", options=[0.5, 1.0, 2.0, 5.0], value=1.0,
            help="Un paso menor produce una curva más fina, pero requiere más cálculos.",
        )
        nominal_flow = st.number_input(
            "Caudal nominal de entrada (L/min)", min_value=1.0, max_value=100.0, value=45.0, step=1.0,
            help="Punto de operación alrededor del cual actúa el controlador.",
        )
        disturbance_start = st.number_input(
            "Inicio de perturbación (s)", min_value=0.0, max_value=float(duration),
            value=min(300.0, float(duration) / 3.0), step=10.0,
        )
        disturbance_step = st.number_input(
            "Cambio de caudal (L/min)", min_value=-40.0, max_value=40.0, value=10.0, step=1.0,
            help="Perturbación escalón aplicada a la alimentación de la celda 1.",
        )
        st.divider()
        st.subheader("Setpoints de nivel")
        sp1 = st.number_input("Celda 1 (cm)", min_value=0.0, max_value=39.0, value=20.0, step=1.0)
        sp2 = st.number_input("Celda 2 (cm)", min_value=0.0, max_value=39.0, value=20.0, step=1.0)
        sp3 = st.number_input("Celda 3 (cm)", min_value=0.0, max_value=39.0, value=20.0, step=1.0)
        compare = st.checkbox("Comparar PI vs. PI + feedforward", value=True)
        st.divider()
        st.caption("Rango del sensor del artículo: 0–39 cm")

    setpoints = (sp1, sp2, sp3)
    initial_levels = setpoints
    with_ff = simulate(
        duration_s=float(duration),
        dt_s=float(dt),
        setpoints_cm=setpoints,
        initial_levels_cm=initial_levels,
        nominal_flow_lpm=float(nominal_flow),
        disturbance_start_s=float(disturbance_start),
        disturbance_step_lpm=float(disturbance_step),
        use_feedforward=True,
    )
    feedback_only = simulate(
        duration_s=float(duration),
        dt_s=float(dt),
        setpoints_cm=setpoints,
        initial_levels_cm=initial_levels,
        nominal_flow_lpm=float(nominal_flow),
        disturbance_start_s=float(disturbance_start),
        disturbance_step_lpm=float(disturbance_step),
        use_feedforward=False,
    )

    if disturbance_start >= duration:
        st.warning("La perturbación comienza al final del ensayo; no habrá respuesta a un escalón visible.")

    st.info(
        "La perturbación se aplica al caudal de alimentación de la celda 1. "
        "El caudal de salida de cada celda alimenta a la siguiente. "
        "Los niveles se limitan al rango del sensor del artículo: 0–39 cm."
    )

    improvement = 100.0 * (1.0 - with_ff.iae_cm_s / np.maximum(feedback_only.iae_cm_s, 1e-12))
    total_iae = float(np.sum(with_ff.iae_cm_s))
    final_error = float(np.max(np.abs(with_ff.errors_cm[-1])))
    peak_error = float(np.max(np.abs(with_ff.errors_cm)))
    average_improvement = float(np.mean(improvement))

    summary_cols = st.columns(4)
    summary_cols[0].metric("IAE total · PI + FF", f"{total_iae:.2f} cm·s")
    summary_cols[1].metric("Error final máximo", f"{final_error:.3f} cm")
    summary_cols[2].metric("Desviación máxima", f"{peak_error:.3f} cm")
    summary_cols[3].metric("Mejora media del IAE", f"{average_improvement:.1f}%" if compare else "—")

    tab_simulation, tab_results, tab_model = st.tabs(["Simulación", "Resultados y datos", "Modelo del artículo"])

    with tab_simulation:
        st.subheader("Vista del proceso")
        st.caption("Estado instantáneo al final del ensayo: tanques, niveles, bombas peristálticas y caudales.")
        st.plotly_chart(
            _process_diagram(with_ff, setpoints),
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        )
        st.subheader("Respuesta de nivel")
        st.caption("Líneas continuas: PI + feedforward · líneas discontinuas: PI solamente · puntos: setpoints")

        fig_levels = go.Figure()
        for index, name in enumerate(CELL_NAMES):
            fig_levels.add_trace(
                go.Scatter(
                    x=with_ff.time_s,
                    y=with_ff.levels_cm[:, index],
                    mode="lines",
                    name=f"{name} · PI + FF",
                    line=dict(color=CELL_COLORS[index], width=2.5),
                    hovertemplate=f"{name} · PI + FF: %{{y:.2f}} cm<extra></extra>",
                )
            )
            if compare:
                fig_levels.add_trace(
                    go.Scatter(
                        x=feedback_only.time_s,
                        y=feedback_only.levels_cm[:, index],
                        mode="lines",
                        name=f"{name} · PI",
                        line=dict(color=CELL_COLORS[index], width=1.5, dash="dash"),
                        hovertemplate=f"{name} · PI: %{{y:.2f}} cm<extra></extra>",
                    )
                )
            fig_levels.add_trace(
                go.Scatter(
                    x=with_ff.time_s,
                    y=np.full(with_ff.time_s.shape, setpoints[index]),
                    mode="lines",
                    name=f"Setpoint · {name}",
                    line=dict(color=CELL_COLORS[index], width=1, dash="dot"),
                    hovertemplate=f"Setpoint · {name}: %{{y:.2f}} cm<extra></extra>",
                    showlegend=False,
                )
            )
        fig_levels.update_layout(
            xaxis_title="Tiempo (s)",
            yaxis_title="Nivel de pulpa (cm)",
            yaxis=dict(range=[0, 39]),
            hovermode="x unified",
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_levels, use_container_width=True, config={"displaylogo": False})

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                _line_chart(
                    with_ff.time_s,
                    {name: with_ff.flows_lpm[:, i] for i, name in enumerate(CELL_NAMES)},
                    y_title="Caudal de cola (L/min)",
                    title="Acciones de control · PI + FF",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
        with col_b:
            st.plotly_chart(
                _line_chart(
                    with_ff.time_s,
                    {name: with_ff.disturbances_lpm[:, i] for i, name in enumerate(CELL_NAMES)},
                    y_title="Caudal de entrada (L/min)",
                    title="Perturbaciones medidas",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )

    with tab_results:
        st.subheader("Indicadores de desempeño")
        metric_tables = [_metrics_table(with_ff, "PI + feedforward")]
        if compare:
            metric_tables.append(_metrics_table(feedback_only, "PI solamente"))
        metrics = pd.concat(metric_tables, ignore_index=True)
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        if compare:
            st.success(
                "Reducción estimada del IAE con feedforward: "
                + ", ".join(f"{CELL_NAMES[i]} {improvement[i]:.1f}%" for i in range(3))
            )

        csv_data = _results_dataframe(with_ff, setpoints).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar resultados PI + feedforward (CSV)",
            data=csv_data,
            file_name="simulacion_control_flotacion.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption("El CSV contiene tiempo, setpoints, niveles, caudales, perturbaciones y errores de las tres celdas.")

    with tab_model:
        st.subheader("Parámetros tomados del artículo")
        st.dataframe(_article_parameters_table(), use_container_width=True, hide_index=True)
        st.markdown(
            r"""
            **Planta integradora con retardo (Tabla 2):**

            \[
            G_{p,i}(s)=\frac{k'_i e^{-\theta_i s}}{s}
            \]

            **Control PI + feedforward (Ecuación 7):**

            \[
            u_i(t)=K_{c,i}\left(e_i(t)+\frac{1}{\tau_{I,i}}
            \int_0^t e_i(\tau)d\tau\right)+K_{ff,i}\Delta Q_{d,i}
            \]

            La simulación usa el caudal nominal como sesgo del actuador para que
            el punto inicial sea estacionario. La perturbación de la celda 1 es
            externa; las celdas 2 y 3 reciben el caudal de salida de la celda anterior.
            """
        )
        with st.expander("Alcance y limitaciones"):
            st.markdown(
                "La app reproduce la capa de control y el modelo reducido del artículo; "
                "no reproduce la química de flotación, la espuma, el PLC, la calibración "
                "de sensores ni los datos experimentales originales."
            )

    st.caption("Fuente: Quintanilla et al., MethodsX 10 (2023) 102081 · DOI 10.1016/j.mex.2023.102081")


if __name__ == "__main__":
    main()

