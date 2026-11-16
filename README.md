# ⚗️ Gemelo Digital SCADA: Control de Nivel en Batería de Flotación

Sistema interactivo de simulación dinámica y supervisión SCADA para el control de nivel de pulpa en un banco de tres celdas de flotación de laboratorio conectadas en serie, implementando **control desacoplado Feedback PI + compensación anticipativa Feedforward**.

Basado rigurosamente en la investigación experimental publicada en:

> **P. Quintanilla, D. Navia, F. Moreno, S. J. Neethling, P. R. Brito-Parada**, *“A methodology to implement a closed-loop feedback-feedforward level control in a laboratory-scale flotation bank using peristaltic pumps”*, **MethodsX** 10 (2023) 102081.  
> DOI: [10.1016/j.mex.2023.102081](https://doi.org/10.1016/j.mex.2023.102081)

---

## 🌟 Características Principales

- **Diagrama P&ID SCADA Interactivo:** Representación gráfica industrial que muestra la pulpa mineral, capa de espuma (*froth*), rebose de concentrado, motores agitadores con eje e impulsor, sensores de nivel LIT ultrasónicos y bombas peristálticas P1–P3 con estado y telemetría de caudal en tiempo real.
- **Modelo Dinámico Realista:** Planta integradora con retardo puro de transporte:
  $$G_{p,i}(s) = \frac{k'_i e^{-\theta_i s}}{s}$$
- **Control Desacoplado con Feedforward:** Comparación directa entre control convencional **PI Feedback** vs. **PI Feedback + Compensación Feedforward**:
  $$u_i(t) = \bar{u}_i + K_{c,i}\left(e_i(t) + \frac{1}{\tau_{I,i}} \int_0^t e_i(\tau)d\tau\right) + K_{ff,i}\left(Q_{d,i}(t) - \bar{Q}_d\right)$$
- **Presets de Escenarios Industriales:** Carga rápida con un clic de ensayos típicos (Caso Base, Perturbación Severa $+25$ L/min, Reducción de Carga $-15$ L/min, Cascada de Setpoints).
- **Métricas Cuantitativas de Control:** Cálculo instantáneo de **IAE** (*Integral Absolute Error*), **SSE** (*Sum of Squared Errors*), error de régimen permanente, tiempos de asentamiento y curva acumulativa de error en el tiempo.
- **Sintonización Avanzada:** Capacidad interactiva de modificar ganancias $K_c$, tiempos integrales $\tau_I$ y factores feedforward $K_{ff}$ por celda.
- **Exportación de Datos:** Descarga de series temporales completas en formato `.csv` para análisis en MATLAB, Python o Excel.
- **Multiplataforma:** Compatible para ejecución local con Streamlit, despliegue estático WebAssembly en **Vercel** vía `stlite`, y contenedores en **Render**.

---

## 📖 Guía de Uso de la Interfaz

### 1. Panel Lateral de Configuración (Sidebar)

1. **Cargar Escenario Rápido:**
   - **🧪 Caso Base (Quintanilla et al. 2023):** Salto escalón de $+10$ L/min en $t=300$ s con setpoints iguales en $20$ cm.
   - **⚡ Perturbación Severa (+25 L/min):** Salto de gran magnitud para evaluar el límite de saturación de bombas.
   - **🔻 Reducción de Caudal (-15 L/min):** Disminución súbita en la alimentación.
   - **🌊 Cascada de Setpoints Desfasados:** Niveles diferenciales ($SP_1=24$ cm, $SP_2=20$ cm, $SP_3=16$ cm).
2. **Parámetros de Operación:**
   - **Duración Total (s):** Tiempo de ensayo virtual (60 a 3600 segundos).
   - **Paso de Integración $\Delta t$ (s):** Resolución numérica (0.25 a 5.0 s).
   - **Caudal Nominal $Q_{d,\text{nom}}$ (L/min):** Punto de equilibrio de flujo inicial (45 L/min por defecto).
   - **Instante del Escalón (s):** Segundo exacto en que ingresa la perturbación en la celda 1.
   - **Salto de Caudal $\Delta Q_d$ (L/min):** Amplitud de la perturbación (positiva o negativa).
3. **Consignas de Nivel (Setpoints):**
   - Configuración individual para Celda 1, Celda 2 y Celda 3 dentro del rango del sensor (0 a 39 cm).
4. **Casilla de Comparación:**
   - Activa/desactiva la visualización simultánea de la curva con solo Feedback PI frente a PI + Feedforward.
5. **🛠️ Sintonización Avanzada (Opcional):**
   - Permite ajustar manualmente $K_c$, $\tau_I$ y $K_{ff}$ para fines pedagógicos o de investigación.

---

### 2. Pestañas de la Aplicación

#### 🖥️ Monitor SCADA & Curvas
- **Diagrama P&ID:** Muestra el estado del banco al finalizar la prueba: nivel en cm, desviación respecto al setpoint, caudal de entrada/salida de cada bomba peristáltica y estado del proceso.
- **Tendencias de Nivel:** Gráfico dinámico interactivo con banda sombreada de tolerancia ($\pm 0.5$ cm) alrededor del setpoint, indicador vertical del evento de perturbación y tooltips al pasar el cursor.

#### 📊 Análisis Comparativo (PI vs PI+FF)
- **Evolución del IAE Acumulado:** Gráfica de $\int_0^t |e(\tau)| d\tau$ que evidencia cómo la compensación anticipativa suprime el error acumulado y evita la propagación del transitorio a las celdas aguas abajo.
- **Tabla de Métricas:** IAE, SSE, Desviación Máxima, Error Final y Tiempo de Asentamiento por celda.
- **Descargar CSV:** Exporta las series completas con un clic.

#### 🎛️ Diagnóstico de Bombas y Actuadores
- Gráfico de caudales $Q_1, Q_2, Q_3$ con líneas guía de límites físicos de bombeo ($Q_{\min}=0$ L/min, $Q_{\max}=100$ L/min).
- Visualización de la perturbación transmitida a cada celda y errores dinámicos individuales.

#### 📚 Modelo Matemático & Metodología
- Deducción matemática, función de transferencia, ecuación de control, parámetros experimentales de la Tabla 2 y enlaces al artículo original.

---

## 💻 Ejecución Local

### Requisitos Previos
- Python 3.10 o superior instalado.

### Paso a Paso

1. Clonar o abrir la carpeta del repositorio:
   ```bash
   cd flotation_control_app
   ```

2. Crear y activar un entorno virtual:
   - **En Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **En Windows (CMD):**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```
   - **En Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Instalar las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

4. Iniciar la aplicación interactiva:
   ```bash
   streamlit run app.py
   ```

5. La aplicación abrirá automáticamente en tu navegador predeterminado en `http://localhost:8501`.

---

## 🚀 Despliegue en Vercel

Este proyecto está configurado para desplegarse **directamente en Vercel** gracias al motor WebAssembly **stlite** (`@stlite/browser`). 

### ¿Cómo funciona en Vercel?
A diferencia de los servidores tradicionales, la arquitectura `stlite` compila Python y las librerías científicas (`numpy`, `pandas`, `plotly`) en WebAssembly (Wasm) dentro del navegador del usuario:
- **Coste cero y sin límites de servidor:** No requiere un backend dedicado ni sufre de desconexión por WebSockets en funciones serverless.
- **Arranque instantáneo:** El archivo `vercel.json` y el punto de entrada estático `index.html` entregan la aplicación a través de la red global Edge CDN de Vercel.

### Pasos para Desplegar en Vercel:

#### Opción A: Despliegue Automático vía GitHub (Recomendado)
1. Sube los cambios a tu repositorio de GitHub:
   ```bash
   git add .
   git commit -m "Mejora visual SCADA y soporte de despliegue en Vercel"
   git push origin main
   ```
2. Entra en tu panel de control de [Vercel](https://vercel.com).
3. Haz clic en **Add New...** -> **Project**.
4. Selecciona tu repositorio `flotation-level-control`.
5. Vercel detectará automáticamente la configuración de `vercel.json` y el archivo `index.html`.
6. Haz clic en **Deploy**. ¡Tu aplicación estará en línea en segundos en una URL `*.vercel.app`!

#### Opción B: Despliegue mediante Vercel CLI
Si tienes instalado Vercel CLI en tu terminal:
```bash
vercel --prod
```

---

## ☁️ Despliegue en Render

El archivo `render.yaml` incluido permite desplegar la versión Python tradicional de Streamlit:

1. Conecta tu repositorio de GitHub en [Render](https://render.com).
2. Crea un nuevo **Web Service** usando el repositorio.
3. Render aplicará la configuración de `render.yaml`:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 📁 Estructura del Proyecto

```text
flotation_control_app/
├── app.py              # Aplicación principal Streamlit y gemelo digital SCADA
├── index.html          # Host estático WebAssembly (stlite) para despliegue en Vercel
├── vercel.json         # Configuración del proyecto y rutas para Vercel
├── .vercelignore       # Exclusión de archivos innecesarios en el build de Vercel
├── render.yaml         # Configuración de despliegue en Render Cloud
├── requirements.txt    # Dependencias de Python (numpy, pandas, plotly, streamlit)
├── .gitignore          # Archivos excluidos del control de versiones
└── README.md           # Guía completa de uso, teoría y despliegue
```

---

## 🔬 Referencia Académica

Si utilizas este simulador o su metodología con fines docentes o de investigación, por favor cita el artículo original:

```bibtex
@article{QUINTANILLA2023102081,
  title = {A methodology to implement a closed-loop feedback-feedforward level control in a laboratory-scale flotation bank using peristaltic pumps},
  journal = {MethodsX},
  volume = {10},
  pages = {102081},
  year = {2023},
  issn = {2215-0161},
  doi = {https://doi.org/10.1016/j.mex.2023.102081},
  author = {Paulina Quintanilla and Daniel Navia and Felipe Moreno and Stephen J. Neethling and Pablo R. Brito-Parada}
}
```
