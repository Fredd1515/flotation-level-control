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


def main() -> None:
    st.title("Control de nivel en una batería de flotación")
    st.caption(
        "Simulación Python del caso de Quintanilla et al. (MethodsX, 2023): "
        "tres celdas en serie, proceso integrador, retardos y PI + feedforward."
    )

    with st.sidebar:
        st.header("Parámetros de la simulación")
        duration = st.number_input("Duración (s)", min_value=60.0, max_value=3600.0, value=900.0, step=60.0)
        dt = st.select_slider("Paso de integración (s)", options=[0.5, 1.0, 2.0, 5.0], value=1.0)
        nominal_flow = st.number_input("Caudal nominal de entrada (L/min)", min_value=1.0, max_value=100.0, value=45.0, step=1.0)
        disturbance_start = st.number_input("Inicio de perturbación (s)", min_value=0.0, max_value=float(duration), value=min(300.0, float(duration) / 3.0), step=10.0)
        disturbance_step = st.number_input("Cambio de caudal (L/min)", min_value=-40.0, max_value=40.0, value=10.0, step=1.0)
        st.divider()
        st.subheader("Setpoints de nivel")
        sp1 = st.number_input("Celda 1 (cm)", min_value=0.0, max_value=39.0, value=20.0, step=1.0)
        sp2 = st.number_input("Celda 2 (cm)", min_value=0.0, max_value=39.0, value=20.0, step=1.0)
        sp3 = st.number_input("Celda 3 (cm)", min_value=0.0, max_value=39.0, value=20.0, step=1.0)
        compare = st.checkbox("Comparar feedback vs. feedback-feedforward", value=True)

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

    st.info(
        "La perturbación se aplica al caudal de alimentación de la celda 1. "
        "El caudal de salida de cada celda alimenta a la siguiente. "
        "Los niveles se limitan al rango del sensor del artículo: 0–39 cm."
    )

    level_series: Dict[str, np.ndarray] = {}
    for i, name in enumerate(CELL_NAMES):
        level_series[f"{name} · PI + FF"] = with_ff.levels_cm[:, i]
        if compare:
            level_series[f"{name} · PI"] = feedback_only.levels_cm[:, i]

    fig_levels = go.Figure()
    for index, name in enumerate(CELL_NAMES):
        fig_levels.add_trace(
            go.Scatter(
                x=with_ff.time_s,
                y=with_ff.levels_cm[:, index],
                mode="lines",
                name=f"{name} · PI + FF",
                line=dict(color=CELL_COLORS[index], width=2),
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
                )
            )
        fig_levels.add_hline(y=setpoints[index], line_color=CELL_COLORS[index], line_width=1, line_dash="dot")
    fig_levels.update_layout(
        title="Respuesta de nivel",
        xaxis_title="Tiempo (s)",
        yaxis_title="Nivel de pulpa (cm)",
        yaxis=dict(range=[0, 39]),
        hovermode="x unified",
        height=480,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_levels, use_container_width=True)

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
        )

    st.subheader("Indicadores de desempeño")
    metric_tables = [_metrics_table(with_ff, "PI + feedforward")]
    if compare:
        metric_tables.append(_metrics_table(feedback_only, "PI solamente"))
    metrics = pd.concat(metric_tables, ignore_index=True)
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    if compare:
        improvement = 100.0 * (1.0 - with_ff.iae_cm_s / np.maximum(feedback_only.iae_cm_s, 1e-12))
        st.success(
            "Reducción estimada del IAE con feedforward: "
            + ", ".join(f"{CELL_NAMES[i]} {improvement[i]:.1f}%" for i in range(3))
        )

    with st.expander("Modelo matemático implementado"):
        st.markdown(
            r"""
            **Planta integradora con retardo (Tabla 2):**

            \[
            G_{p,i}(s)=\frac{k'_i e^{-\theta_i s}}{s},
            \qquad k'_i=-0.001018,\quad \theta=[7,6,3]\ \mathrm{s}
            \]

            **Control PI + feedforward (Ecuación 7):**

            \[
            u_i(t)=K_{c,i}\left(e_i(t)+\frac{1}{\tau_{I,i}}
            \int_0^t e_i(\tau)d\tau\right)+K_{ff,i}\Delta Q_{d,i}
            \]

            con \(K_c=[-62,-76,-85]\), \(\tau_I=[56,30,44]\) s y
            \(K_{ff}=[0.8,0.8,1.0]\). La simulación añade el caudal nominal
            como sesgo del actuador para que el punto inicial sea estacionario.
            """
        )

    with st.expander("Alcance y limitaciones"):
        st.markdown(
            "La app reproduce la capa de control y el modelo reducido del artículo; "
            "no reproduce la química de flotación, la espuma, el PLC, la calibración "
            "de sensores ni los datos experimentales originales. Los parámetros de "
            "las bombas y sensores se pueden sustituir por datos propios si el docente "
            "los solicita."
        )

    st.caption("Fuente: Quintanilla et al., MethodsX 10 (2023) 102081, doi:10.1016/j.mex.2023.102081")


if __name__ == "__main__":
    main()

