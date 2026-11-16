# Simulación de control de nivel en flotación

Interfaz gráfica en Python/Streamlit basada en el artículo:

> P. Quintanilla, D. Navia, F. Moreno et al., “A methodology to implement a closed-loop feedback-feedforward level control in a laboratory-scale flotation bank using peristaltic pumps”, MethodsX 10 (2023) 102081. doi: [10.1016/j.mex.2023.102081](https://doi.org/10.1016/j.mex.2023.102081)

## Qué simula

- Tres celdas de flotación conectadas en serie.
- Modelo integrador con retardo: `Gp_i(s) = k'_i exp(-theta_i s) / s`.
- Control PI con los parámetros finales de la Tabla 2.
- Feedforward con `Kff = [0.8, 0.8, 1.0]`, según las ecuaciones 7 y 8.
- Comparación entre PI solamente y PI + feedforward.
- Perturbación del caudal de alimentación y métricas IAE/SSE.

## Ejecutar localmente

Con Python 3.10 o superior:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

## Interpretación

El modelo representa la capa de control y el modelo reducido del artículo. No pretende reemplazar el montaje experimental: no incluye PLC, sensores físicos, calibración, química de reactivos, espuma ni datos experimentales originales. El sesgo de caudal nominal se incluye para mantener el punto de operación inicial estacionario.

## Despliegue en Render

El archivo `render.yaml` deja configurado el servicio web. El repositorio debe estar en GitHub, GitLab o Bitbucket; luego se puede crear el servicio desde Render usando ese repositorio. Render instalará `requirements.txt` y ejecutará la app con el puerto público asignado.

