import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

st.set_page_config(page_title="RNN: precio de una acción", layout="wide")
BASE = Path(__file__).parent
plt.rcParams["figure.dpi"] = 110


# ------------------------------------------------------------------
# Carga del modelo y la escala generados en el Colab
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(huella, _datos):
    """Keras necesita una ruta: se escribe el archivo en una carpeta temporal."""
    from tensorflow import keras
    ruta = Path(tempfile.gettempdir()) / f"modelo_{huella}.keras"
    if not ruta.exists():
        ruta.write_bytes(_datos)
    return keras.models.load_model(ruta)


with st.sidebar:
    st.header("Modelo del Colab")
    st.caption("Sube los archivos de la sección 6. Despliegue. Si no subes nada, "
               "se usan los que están en el repositorio.")
    up_modelo = st.file_uploader("modelo_rnn.keras", type=["keras"])
    up_escala = st.file_uploader("escala.json", type=["json"])

if up_modelo is not None and up_escala is not None:
    datos_modelo = up_modelo.getvalue()
    escala = json.loads(up_escala.getvalue())
    origen = "archivos subidos"
elif (BASE / "modelo_rnn.keras").exists() and (BASE / "escala.json").exists():
    datos_modelo = (BASE / "modelo_rnn.keras").read_bytes()
    escala = json.loads((BASE / "escala.json").read_text())
    origen = "repositorio"
else:
    st.title("Redes neuronales recurrentes: el precio de una acción")
    st.warning(
        "Sube **modelo_rnn.keras** y **escala.json** en la barra lateral "
        "(los descargas del Colab, sección 6. Despliegue)."
    )
    st.stop()

faltan = [k for k in ("p_min", "p_max", "ventana") if k not in escala]
if faltan:
    st.error("El archivo escala.json no tiene: " + ", ".join(faltan))
    st.stop()

HUELLA = hashlib.md5(datos_modelo).hexdigest()[:10]
P_MIN, P_MAX, VENTANA = escala["p_min"], escala["p_max"], int(escala["ventana"])
try:
    with st.spinner("Cargando el modelo entrenado en el Colab…"):
        modelo = cargar_modelo(HUELLA, datos_modelo)
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()
st.sidebar.success(f"Modelo cargado desde: {origen}")
UNIDADES = modelo.layers[0].units

escalar = lambda v: (np.asarray(v) - P_MIN) / (P_MAX - P_MIN)
desescalar = lambda v: np.asarray(v) * (P_MAX - P_MIN) + P_MIN


# ------------------------------------------------------------------
# Los mismos datos del Colab (misma semilla)
# ------------------------------------------------------------------
@st.cache_data
def datos():
    np.random.seed(42)
    retornos = np.random.normal(loc=0.0008, scale=0.015, size=300)
    return 100 * np.exp(np.cumsum(retornos))


precios = datos()
CORTE = int(len(precios) * 0.8)
serie = escalar(precios)
X = np.array([serie[i:i + VENTANA] for i in range(len(serie) - VENTANA)])[..., np.newaxis]
y = np.array([serie[i + VENTANA] for i in range(len(serie) - VENTANA)])
N_TRAIN = CORTE - VENTANA
X_test, y_test = X[N_TRAIN:], y[N_TRAIN:]


@st.cache_data
def evaluar(huella):
    est = desescalar(modelo.predict(X_test, verbose=0).ravel())
    real = desescalar(y_test)
    base = desescalar(X_test[:, -1, 0])
    return est, real, base


def que_probar(lista):
    st.info("**Qué probar**\n\n" + "\n".join(f"- {q}" for q in lista))


def cierre(conclusion, pregunta):
    st.success("**Conclusión**\n\n" + conclusion)
    st.markdown(f"**Pregunta:** {pregunta}")


def dibujar_rnn(unidades=8, ventana=5):
    TEAL, CORAL, GRIS = "#1D9E75", "#D85A30", "#888780"
    VERDE_CLARO, CORAL_CLARO = "#CBEBDD", "#F5D3C5"
    p_rnn = unidades + unidades * unidades + unidades
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [1, 1]})
    def flecha(ax, a, b, c="black", **k):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=c, **k))

    # ============ 1. La red como se construye ============
    ax = ax1
    ys = [0.5] if unidades == 1 else [0.85 - i * 0.7 / (unidades - 1) for i in range(unidades)]
    xin, xh, xd, xo = 0.12, 0.42, 0.72, 0.9
    for y in ys:
        ax.plot([xin, xh], [0.5, y], color=GRIS, lw=0.8, alpha=0.6, zorder=1)
        ax.plot([xh, xd], [y, 0.5], color=CORAL, lw=1, alpha=0.7, ls="--", zorder=1)
    ax.add_patch(FancyBboxPatch((xh - 0.05, 0.08), 0.1, 0.84, boxstyle="round,pad=0.01", fc="none", ec=TEAL, ls="--"))
    ax.scatter([xh] * unidades, ys, s=380, c=VERDE_CLARO, edgecolors=TEAL, zorder=2)
    for y in ys:
        ax.text(xh, y, "n", ha="center", va="center", fontsize=8, zorder=3)
    ax.scatter([xin], [0.5], s=1200, c="lightgray", zorder=2)
    ax.text(xin, 0.5, "x", ha="center", va="center", fontsize=15)
    ax.scatter([xd], [0.5], s=1000, c=CORAL_CLARO, edgecolors=CORAL, zorder=2)
    ax.text(xd, 0.5, "Dense", ha="center", va="center", fontsize=9)
    flecha(ax, (xd + 0.035, 0.5), (xo - 0.03, 0.5))
    ax.text(xo, 0.5, "ŷ", ha="center", va="center", fontsize=16, bbox=dict(boxstyle="round", fc="lightgray", ec="gray"))
    ax.add_patch(FancyArrowPatch((xh + 0.035, 0.94), (xh - 0.035, 0.94), connectionstyle="arc3,rad=1.3",
                                 arrowstyle="-|>", color=TEAL, lw=2, mutation_scale=14))
    ax.text(xh + 0.07, 1.03, f"La salida h ({unidades} números) se guarda\ny vuelve como memoria al día siguiente",
            fontsize=10, color=TEAL, va="center")
    ax.text(xd, 0.72, "solo después\ndel último día", ha="center", fontsize=10, color=CORAL)
    ax.text(xin, 0.36, "precio de UN día\n(entran de a uno)", ha="center", va="top", fontsize=10)
    ax.text(xin, -0.04, "Entrada", ha="center", va="top", fontsize=11, weight="bold")
    ax.text(xh, -0.04, f"SimpleRNN({unidades}): {unidades} neuronas\n{unidades} + {unidades*unidades} + {unidades} = {p_rnn} pesos",
            ha="center", va="top", fontsize=11)
    ax.text(xd, -0.04, f"Dense(1)\n{unidades} + 1 = {unidades+1} pesos", ha="center", va="top", fontsize=11)
    ax.set_title("1. La red como se construye: una sola capa de neuronas que se reutiliza", fontsize=13, loc="left")

    # ============ 2. La misma capa, día por día ============
    ax = ax2
    xs = [0.07 + i * 0.62 / max(ventana - 1, 1) for i in range(ventana)]
    w = min(0.1, 0.5 / ventana)
    fs = 10 if ventana <= 8 else 7
    n_dib = min(unidades, 8)
    cy = [0.575] if n_dib == 1 else [0.7 - i * 0.26 / (n_dib - 1) for i in range(n_dib)]
    for i, x in enumerate(xs):
        ultimo = i == ventana - 1
        ax.add_patch(FancyBboxPatch((x - w / 2, 0.36), w, 0.43, boxstyle="round,pad=0.01",
                                    fc=VERDE_CLARO if not ultimo else "#A6DCC4", ec=TEAL, lw=1.2 if not ultimo else 2.2))
        ax.text(x, 0.755, f"Día {i+1}", ha="center", va="center", fontsize=fs, weight="bold")
        ax.scatter([x] * n_dib, cy, s=45 if ventana <= 8 else 15, c="white", edgecolors=TEAL, zorder=3)
        if unidades > 8:
            ax.text(x, 0.4, "…", ha="center", va="center", fontsize=fs)
        ax.scatter([x], [0.16], s=550 if ventana <= 8 else 200, c="lightgray", zorder=2)
        ax.text(x, 0.16, f"x{i+1}", ha="center", va="center", fontsize=fs - 1, zorder=3)
        flecha(ax, (x, 0.2), (x, 0.345), c="gray")
        if not ultimo:
            flecha(ax, (x + w / 2 + 0.008, 0.55), (xs[i+1] - w / 2 - 0.008, 0.55), c=TEAL, lw=2)
            ax.text((x + xs[i+1]) / 2, 0.59, f"h{i+1}", ha="center", fontsize=fs - 1, color=TEAL, weight="bold")
    xd = xs[-1] + 0.14
    flecha(ax, (xs[-1] + w / 2 + 0.008, 0.55), (xd - 0.035, 0.55), c=CORAL, lw=2)
    ax.text((xs[-1] + xd) / 2, 0.59, f"h{ventana}", ha="center", fontsize=fs - 1, color=CORAL, weight="bold")
    ax.scatter([xd], [0.55], s=1000, c=CORAL_CLARO, edgecolors=CORAL, zorder=2)
    ax.text(xd, 0.55, "Dense", ha="center", va="center", fontsize=9, zorder=3)
    flecha(ax, (xd + 0.035, 0.55), (xd + 0.1, 0.55))
    ax.text(xd + 0.13, 0.55, "ŷ", ha="center", va="center", fontsize=16, bbox=dict(boxstyle="round", fc="lightgray", ec="gray"))
    ax.text(xd + 0.13, 0.44, f"precio estimado\ndel día {ventana+1}", ha="center", va="top", fontsize=10)
    # llave sobre las cajas
    y0 = 0.85
    ax.plot([xs[0] - w / 2, xs[-1] + w / 2], [y0, y0], color=TEAL, lw=1.5)
    for xx in (xs[0] - w / 2, xs[-1] + w / 2):
        ax.plot([xx, xx], [y0, y0 - 0.03], color=TEAL, lw=1.5)
    ax.text((xs[0] + xs[-1]) / 2, y0 + 0.04,
            f"Son las MISMAS {unidades} neuronas (los mismos {p_rnn} pesos) en cada día — no son {ventana*unidades} neuronas distintas",
            ha="center", fontsize=10.5, color=TEAL)
    ax.text(0.0, -0.02,
            f"Días 1 a {ventana-1}: la capa solo produce la memoria h y se la pasa al día siguiente. No hay predicción todavía.\n"
            f"Día {ventana}: el último h es el único que llega a Dense, que produce la única predicción ŷ.",
            ha="left", va="top", fontsize=11)
    ax.set_title("2. La misma red vista en el tiempo: un cuadro por día", fontsize=13, loc="left")

    for a in (ax1, ax2):
        a.set_xlim(-0.02, 1.08); a.set_ylim(-0.15, 1.1); a.axis("off")
    plt.tight_layout()
    return fig



# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
st.title("Redes neuronales recurrentes: el precio de una acción")
st.caption("Compañera del Colab. El modelo de las pestañas 4 y 5 es el que entrenaste y guardaste en el Colab. "
           "Es un ejemplo educativo: no sirve para decisiones de inversión.")

t1, t2, t3, t4, t5 = st.tabs([
    "1. Datos y ventanas", "2. Una neurona", "3. Estructura de la red",
    "4. Evaluación", "5. Estimar el día siguiente",
])

# ---------------- 1. Datos y ventanas ----------------
with t1:
    st.markdown(
        f"Son {len(precios)} días de precios simulados, los mismos del Colab. Los {CORTE} primeros son para "
        f"entrenar y los {len(precios) - CORTE} últimos para probar. Cada ejemplo es una **ventana** de "
        f"{VENTANA} días con su respuesta: el día siguiente."
    )
    inicio = st.slider("Mueve la ventana (día en que empieza)", 1, len(precios) - VENTANA, 1)
    i0 = inicio - 1
    fig, ax = plt.subplots(figsize=(11, 3.8))
    dias = np.arange(1, len(precios) + 1)
    ax.plot(dias, precios, color="#1F4E8C", lw=1.3)
    ax.axvspan(CORTE + 0.5, len(precios) + 0.5, color="gray", alpha=0.12, label="prueba")
    ax.axvspan(inicio - 0.5, inicio + VENTANA - 0.5, color="#FFD166", alpha=0.6, label=f"ventana ({VENTANA} días)")
    ax.scatter(dias[i0:i0 + VENTANA], precios[i0:i0 + VENTANA], color="#1F4E8C", zorder=3, s=20)
    ax.scatter([inicio + VENTANA], [precios[i0 + VENTANA]], color="#D85A30", zorder=3, s=45, label="respuesta")
    ax.set_xlabel("Día"); ax.set_ylabel("Precio ($)"); ax.grid(alpha=0.3); ax.legend(loc="upper left")
    st.pyplot(fig); plt.close(fig)

    st.markdown(
        f"Esta ventana usa los días **{inicio} a {inicio + VENTANA - 1}** para estimar el día **{inicio + VENTANA}** "
        f"(${precios[i0 + VENTANA]:.2f})."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Días de entrenamiento", CORTE)
    c2.metric("Ventanas de entrenamiento", f"{CORTE} − {VENTANA} = {N_TRAIN}")
    c3.metric("Ventanas de prueba", len(X_test))
    st.markdown(
        f"Las ventanas **se deslizan de a un día** y se traslapan; no son bloques separados. Por eso la forma de "
        f"los datos es `({N_TRAIN}, {VENTANA}, 1)`: {N_TRAIN} ventanas, {VENTANA} días cada una, 1 variable (el precio)."
    )
    que_probar([
        "Mueve la ventana de a un día y fíjate cuántos días comparte con la anterior.",
        "Lleva la ventana hasta el final de la zona blanca: ¿dónde queda su respuesta?",
    ])
    cierre(
        "Una red no recibe la serie completa: la serie se corta en **ventanas** que se deslizan de a un día. "
        "Cada ventana es una pregunta (los precios de varios días) y su respuesta es el precio del día siguiente. "
        "Al deslizarlas se obtienen muchos ejemplos con pocos datos. La parte de prueba va **al final** "
        "porque así se simula lo que pasa en la realidad: aprender del pasado para estimar el futuro.",
        f"Si hubiera 100 días de entrenamiento y ventanas de {VENTANA} días, ¿cuántas ventanas saldrían?",
    )

# ---------------- 2. Una neurona ----------------
with t2:
    st.markdown(
        "Es la red de la app web y de la parte A del Colab: **una sola neurona**, 12 precios y ventanas de 3 días."
    )
    st.latex(r"h_t = \tanh(w_x x_t + w_h h_{t-1} + b) \qquad \hat{y} = w_y h_3 + b_y")
    mini = np.array([42, 43.5, 44, 46, 45.2, 47, 48.5, 48, 50, 51.5, 51, 53])
    mn, mx = mini.min(), mini.max()
    nm = (mini - mn) / (mx - mn)
    Xm = np.array([nm[i:i + 3] for i in range(9)]); ym = nm[3:]
    DEF = {"wx": 0.5, "wh": 0.3, "b": 0.0, "wy": 0.8, "by": 0.0}
    for k, v in DEF.items():
        st.session_state.setdefault("a_" + k, v)

    def restablecer():
        for k, v in DEF.items():
            st.session_state["a_" + k] = v

    cols = st.columns(5)
    etiquetas = {"wx": "w_x (entrada)", "wh": "w_h (memoria)", "b": "b (sesgo)", "wy": "w_y (salida)", "by": "b_y (sesgo salida)"}
    p = {k: cols[i].slider(etiquetas[k], -2.0, 2.0, step=0.01, key="a_" + k) for i, k in enumerate(DEF)}
    st.button("Restablecer pesos", on_click=restablecer)

    def adelante(x):
        h = [0.0]
        for t in range(3):
            h.append(np.tanh(p["wx"] * x[t] + p["wh"] * h[-1] + p["b"]))
        return p["wy"] * h[3] + p["by"], h

    v = st.selectbox("Ventana", range(9), format_func=lambda k: f"{k + 1}: precios {mini[k:k + 3].tolist()} → real {mini[k + 3]}")
    yv, h = adelante(Xm[v])
    lineas = [f"h{t} = tanh({p['wx']:.2f}·{Xm[v][t-1]:.3f} + {p['wh']:.2f}·{h[t-1]:.3f} + {p['b']:.2f}) = {h[t]:.3f}" for t in (1, 2, 3)]
    lineas.append(f"ŷ  = {p['wy']:.2f}·{h[3]:.3f} + {p['by']:.2f} = {yv:.3f}  →  ${yv * (mx - mn) + mn:.2f}  (real ${mini[v + 3]:.2f})")
    st.code("\n".join(lineas), language=None)

    est = [adelante(x)[0] * (mx - mn) + mn for x in Xm]
    mse = np.mean([(adelante(x)[0] - t) ** 2 for x, t in zip(Xm, ym)])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(range(1, 13), mini, marker="o", label="Real", color="#1F4E8C")
    ax.plot(range(4, 13), est, marker="o", ls="--", label="Estimación de la neurona", color="#D85A30")
    ax.set_xlabel("Día"); ax.set_ylabel("Precio ($)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"MSE (escala 0–1): {mse:.4f}")
    st.pyplot(fig); plt.close(fig)
    que_probar([
        "Pon w_h en 0: la neurona pierde la memoria y solo mira el último día.",
        "Intenta bajar el MSE a mano moviendo los deslizadores. Con entrenamiento se llega a 0.0045.",
        "Pon w_x en 0: ¿qué información le queda a la neurona?",
    ])
    cierre(
        "La neurona recurrente lee **un día a la vez** y guarda en **h** un resumen de lo que ha visto: esa es su "
        "memoria. En cada día combina el precio nuevo con la memoria del día anterior, siempre con **los mismos pesos**. "
        "Encontrar buenos pesos a mano es difícil; el entrenamiento lo hace de forma automática.",
        "¿Qué pasa con la memoria de la neurona cuando w_h vale 0?",
    )

# ---------------- 3. Estructura ----------------
with t3:
    st.markdown(
        f"El modelo del Colab tiene **{UNIDADES} neuronas** recurrentes, ventana de **{VENTANA} días** y "
        f"**{modelo.count_params()} parámetros**. Cambia los valores para ver cómo sería la red con otra configuración."
    )
    c1, c2 = st.columns(2)
    n = c1.slider("Neuronas de la capa recurrente", 1, 32, UNIDADES)
    vent = c2.slider("Tamaño de la ventana (días)", 3, 20, VENTANA)
    fig = dibujar_rnn(n, vent)
    st.pyplot(fig); plt.close(fig)
    tabla = pd.DataFrame({
        "Parte": ["w_x (precio → neuronas)", "w_h (memoria de ayer → neuronas de hoy)", "b (sesgos)",
                  "Dense: pesos", "Dense: sesgo", "Total"],
        "Cálculo": [f"{n} × 1", f"{n} × {n}", f"{n}", f"{n}", "1", ""],
        "Parámetros": [n, n * n, n, n, 1, n + n * n + n + n + 1],
    })
    st.dataframe(tabla, hide_index=True, width="stretch")
    que_probar([
        "Pon 1 neurona y 3 días: es exactamente la red de la pestaña 2 (5 parámetros).",
        "Pasa de 8 a 16 neuronas: ¿cuánto crecen los parámetros?",
        "Cambia la ventana de 5 a 20 días: ¿cambia el total de parámetros?",
    ])
    cierre(
        "Una RNN es **una sola capa de neuronas que se reutiliza** en cada día de la ventana. Por eso la cantidad "
        "de parámetros depende del número de neuronas, pero **no del tamaño de la ventana**. Durante la ventana la "
        "capa solo va actualizando su memoria, y únicamente la memoria del **último día** pasa a la capa Dense, "
        "que da la estimación.",
        "Si la ventana pasa de 5 a 20 días, ¿cambia la cantidad de parámetros?",
    )

# ---------------- 4. Evaluación ----------------
with t4:
    est, real, base = evaluar(HUELLA)
    mae_rnn = np.mean(np.abs(est - real)); mae_base = np.mean(np.abs(base - real))
    st.markdown(
        f"Comparamos el modelo con una **línea base** muy simple: *el precio de mañana será igual al de hoy*. "
        f"Los datos son los {len(real)} días de prueba, que el modelo nunca vio al entrenar."
    )
    c1, c2 = st.columns(2)
    c1.metric("Error promedio de la RNN (MAE)", f"${mae_rnn:.2f}")
    c2.metric("Error promedio de la línea base (MAE)", f"${mae_base:.2f}")
    ver_base = st.checkbox("Mostrar también la línea base en el gráfico")
    dias_t = np.arange(CORTE + 1, len(precios) + 1)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(dias_t, real, label="Real", color="#1F4E8C")
    ax.plot(dias_t, est, "--", label="Estimación RNN", color="#D85A30")
    if ver_base:
        ax.plot(dias_t, base, ":", label="Línea base (mañana = hoy)", color="gray")
    ax.set_xlabel("Día"); ax.set_ylabel("Precio ($)"); ax.grid(alpha=0.3); ax.legend()
    st.pyplot(fig); plt.close(fig)
    que_probar([
        "Activa la línea base y compárala con la RNN: ¿se parecen?",
        "Fíjate en los días con saltos grandes: ¿la RNN los anticipa o llega tarde?",
    ])
    ganador = "la RNN" if mae_rnn < mae_base else "la línea base"
    cierre(
        f"En este modelo el menor error lo tuvo **{ganador}** (${min(mae_rnn, mae_base):.2f} frente a "
        f"${max(mae_rnn, mae_base):.2f}). Los precios de este ejemplo cambian al azar cada día, así que no hay un "
        "patrón que aprender: la mejor estimación es casi el precio de hoy. Por eso la RNN termina copiando el último "
        "precio y su curva parece ir **un día detrás** de la real. La lección: un modelo solo sirve si le gana a una "
        "línea base sencilla.",
        "¿Cuál tuvo menor error: la RNN o la línea base?",
    )

# ---------------- 5. Estimar ----------------
with t5:
    st.markdown(f"Escribe los precios de **{VENTANA} días seguidos** y el modelo del Colab calcula una estimación para el día siguiente.")
    cols = st.columns(VENTANA)
    entradas = [cols[i].number_input(f"Día {i + 1}", value=float(round(precios[-VENTANA + i], 2)), step=0.5, format="%.2f")
                for i in range(VENTANA)]
    x = escalar(entradas).reshape(1, VENTANA, 1)
    valor = float(desescalar(modelo.predict(x, verbose=0)[0, 0]))
    st.metric("Estimación para el día siguiente", f"${valor:.2f}", f"{valor - entradas[-1]:+.2f} frente al último día")
    if max(entradas) > P_MAX or min(entradas) < P_MIN:
        st.warning(
            f"Algunos precios están fuera del rango del entrenamiento (${P_MIN:.2f} a ${P_MAX:.2f}). "
            "El modelo nunca vio valores así, y su estimación puede ser poco confiable."
        )
    que_probar([
        "Escribe una caída fuerte, por ejemplo 120, 118, 115, 111, 106: ¿el modelo sigue la tendencia?",
        "Escribe cinco precios iguales: ¿qué estimación da?",
        "Escribe precios de 200 o de 20: ¿qué pasa fuera del rango de entrenamiento?",
    ])
    cierre(
        "El modelo estima valores **cercanos al último precio**, con un ajuste pequeño según la tendencia, porque "
        "eso fue lo que aprendió de los datos. Además, solo es confiable **dentro del rango de precios del "
        f"entrenamiento** (${P_MIN:.2f} a ${P_MAX:.2f}): los precios se escalan siempre con ese mismo mínimo y "
        "máximo, y lo que queda por fuera es algo que el modelo nunca vio.",
        "Si escribes cinco precios iguales, ¿qué estimación da el modelo?",
    )
