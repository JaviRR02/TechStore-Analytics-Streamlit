from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Configuración inicial
st.set_page_config(
    page_title="TechStore - Analytics Predictivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"


@st.cache_data
def cargar_csv(nombre):
    return pd.read_csv(DATA / nombre)


@st.cache_data
def cargar_metadata():
    with open(
        DATA / "metadata_app.json",
        encoding="utf-8"
    ) as archivo:
        return json.load(archivo)


@st.cache_resource
def cargar_modelo():
    return joblib.load(
        MODELS / "regresion_polinomial.pkl"
    )


# Recuperar información
metadata = cargar_metadata()

ventas = cargar_csv("ventas_historicas.csv")
forecast = cargar_csv("forecasting.csv")
segmentos = cargar_csv("segmentos.csv")
churn = cargar_csv("churn_segmentos.csv")
alertas = cargar_csv("alertas.csv")
roi = cargar_csv("roi_retencion.csv")
metricas = cargar_csv("metricas_churn.csv")

modelo = cargar_modelo()

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
forecast["fecha"] = pd.to_datetime(forecast["fecha"])

ventas = ventas.sort_values("fecha").reset_index(drop=True)

nombres_cortos = {
    "Alto valor y actividad": "Alto valor",
    "Activos de valor medio": "Valor medio",
    "Baja actividad reciente": "Baja actividad"
}

# Encabezado general
st.title("TechStore - Analytics Predictivo")

st.caption(
    "Sistema integrado de ventas, segmentación, churn "
    "y evaluación económica"
)

with st.sidebar:
    st.header("Información del proyecto")
    st.write("**Módulo:** MGR")
    st.write("**Empresa:** TechStore")
    st.write("**Versión:** 1.0.0")
    st.divider()
    st.info(
        "Aplicación demostrativa con datos históricos. "
        "No utiliza información en tiempo real."
    )

# Vistas por audiencia
tab_ceo, tab_marketing, tab_analistas = st.tabs([
    "Dirección / CEO",
    "Marketing / CMO",
    "Analistas"
])


# --------------------------------------------------
# VISTA 1: DIRECCIÓN
# --------------------------------------------------

with tab_ceo:

    st.header("Resumen ejecutivo")

    kpis = metadata["kpis"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Clientes",
        f"{int(kpis['Clientes analizados']):,}"
    )

    col2.metric(
        "Ingresos transaccionales",
        f"{kpis['Ingresos transaccionales (€)']:,.2f} €"
    )

    col3.metric(
        "Segmentos RFM",
        int(kpis["Segmentos RFM"])
    )

    col4.metric(
        "ROI hipotético de retención",
        f"{kpis['ROI retención hipotético (%)']:.2f}%"
    )

    columna_a, columna_b = st.columns(2)

    with columna_a:

        fig_ventas = go.Figure()

        fig_ventas.add_trace(
            go.Scatter(
                x=ventas["fecha"],
                y=ventas["ventas"],
                mode="lines+markers",
                name="Ventas reales"
            )
        )

        fig_ventas.add_trace(
            go.Scatter(
                x=ventas["fecha"],
                y=ventas["ventas_estimadas"],
                mode="lines",
                name="Estimadas",
                line=dict(dash="dash")
            )
        )

        fig_ventas.update_layout(
            title="Ventas históricas y estimadas",
            xaxis_title="Fecha",
            yaxis_title="Ventas (€)",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_ventas,
            use_container_width=True
        )

    with columna_b:

        fig_forecast = px.line(
            forecast,
            x="fecha",
            y="ventas_proyectadas",
            markers=True,
            title="Proyección retrospectiva de 2024",
            labels={
                "fecha": "Fecha",
                "ventas_proyectadas": "Ventas (€)"
            }
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )

    columna_c, columna_d = st.columns(2)

    with columna_c:

        fig_segmentos = px.bar(
            segmentos,
            x="segmento_rfm",
            y="porcentaje_ingresos",
            color="segmento_rfm",
            title="Participación de ingresos por segmento",
            labels={
                "segmento_rfm": "Segmento",
                "porcentaje_ingresos": "Ingresos (%)"
            }
        )

        fig_segmentos.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_segmentos,
            use_container_width=True
        )

    with columna_d:

        fig_roi = px.bar(
            roi,
            x="segmento_rfm",
            y="roi_estimado_pct",
            color="segmento_rfm",
            title="ROI hipotético de retención",
            labels={
                "segmento_rfm": "Segmento",
                "roi_estimado_pct": "ROI (%)"
            }
        )

        fig_roi.add_hline(
            y=0,
            line_dash="dash"
        )

        fig_roi.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_roi,
            use_container_width=True
        )

    st.warning(
        "El ROI corresponde a escenarios hipotéticos. "
        "No representa un beneficio económico comprobado."
    )


# --------------------------------------------------
# VISTA 2: MARKETING
# --------------------------------------------------

with tab_marketing:

    st.header("Marketing y simulación comercial")

    st.write(
        "Modificar el presupuesto de marketing y el "
        "precio promedio para explorar escenarios "
        "con el modelo de regresión polinomial."
    )

    marketing_pct = st.slider(
        "Cambio en presupuesto de marketing (%)",
        min_value=-10,
        max_value=20,
        value=10,
        step=5
    )

    precio_pct = st.slider(
        "Cambio en precio promedio (%)",
        min_value=-10,
        max_value=10,
        value=-5,
        step=5
    )

    variables = [
        "marketing_budget",
        "precio_promedio",
        "competencia_precio",
        "mes",
        "tendencia"
    ]

    # Escenario base: diciembre 2023
    base = ventas.iloc[[-1]][variables].copy()

    escenario = base.copy()

    escenario["marketing_budget"] *= (
        1 + marketing_pct / 100
    )

    escenario["precio_promedio"] *= (
        1 + precio_pct / 100
    )

    ventas_base = float(
        modelo.predict(base)[0]
    )

    ventas_escenario = float(
        modelo.predict(escenario)[0]
    )

    diferencia = ventas_escenario - ventas_base

    variacion = (
        diferencia / ventas_base * 100
        if ventas_base != 0 else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Ventas base estimadas",
        f"{ventas_base:,.2f} €"
    )

    c2.metric(
        "Ventas del escenario",
        f"{ventas_escenario:,.2f} €"
    )

    c3.metric(
        "Diferencia estimada",
        f"{diferencia:+,.2f} €",
        delta=f"{variacion:+.2f}%"
    )

    df_comparacion = pd.DataFrame({
        "Escenario": ["Base", "Simulado"],
        "Ventas (€)": [
            ventas_base,
            ventas_escenario
        ]
    })

    fig_escenario = px.bar(
        df_comparacion,
        x="Escenario",
        y="Ventas (€)",
        color="Escenario",
        text="Ventas (€)",
        title="Comparación de ventas estimadas"
    )

    fig_escenario.update_traces(
        texttemplate="%{text:,.2f}",
        textposition="outside"
    )

    fig_escenario.update_layout(
        showlegend=False,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_escenario,
        use_container_width=True
    )

    # Advertencias de extrapolación
    entrenamiento = ventas.iloc[:30]

    variables_ajustadas = [
        "marketing_budget",
        "precio_promedio"
    ]

    fuera_rango = []

    for variable in variables_ajustadas:

        valor = escenario[variable].iloc[0]

        minimo = entrenamiento[variable].min()
        maximo = entrenamiento[variable].max()

        if valor < minimo or valor > maximo:
            fuera_rango.append(variable)

    if fuera_rango:
        st.warning(
            "Valores fuera del rango de entrenamiento: "
            + ", ".join(fuera_rango)
        )

    st.info(
        "Referencia: diciembre de 2023. "
        "El escenario se encuentra fuera del periodo "
        "original de entrenamiento y es exploratorio. "
        "Las diferencias no demuestran efectos causales "
        "ni representan un cálculo de ROI."
    )

    st.subheader("Comportamiento de clientes")

    fig_churn = px.bar(
        churn,
        x="segmento_rfm",
        y="tasa_churn_pct",
        color="segmento_rfm",
        title="Churn observado por segmento",
        labels={
            "segmento_rfm": "Segmento",
            "tasa_churn_pct": "Churn (%)"
        }
    )

    fig_churn.update_layout(showlegend=False)

    st.plotly_chart(
        fig_churn,
        use_container_width=True
    )

    st.caption(
        "Las tasas de churn mostradas corresponden "
        "al conjunto histórico de prueba."
    )


# --------------------------------------------------
# VISTA 3: ANALISTAS
# --------------------------------------------------

with tab_analistas:

    st.header("Evaluación técnica de modelos")

    st.subheader("Modelos de clasificación de churn")

    columnas_metricas = [
        "Modelo",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]

    st.dataframe(
        metricas[columnas_metricas],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Indicadores de mantenimiento")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE ventas (prueba)",
        f"{metadata['kpis']['MAE ventas prueba (€)']:,.2f} €"
    )

    m2.metric(
        "ROC-AUC churn",
        f"{metadata['kpis']['ROC-AUC churn prueba']:.4f}"
    )

    m3.metric(
        "Clientes con alertas",
        f"{int(metadata['kpis']['Clientes con alertas']):,}"
    )

    fig_alertas = px.bar(
        alertas,
        x="Prioridad",
        y="Clientes",
        color="Prioridad",
        title="Alertas exploratorias de anomalías"
    )

    fig_alertas.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_alertas,
        use_container_width=True
    )

    st.warning(
        "Limitación conocida: Random Forest no detectó "
        "ninguno de los 10 casos de churn del segmento "
        "de alto valor en el conjunto histórico de prueba."
    )

    st.info(
        "El monitoreo realizado verifica reproducibilidad "
        "histórica. No hay datos nuevos para medir drift "
        "o degradación en producción."
    )

    st.subheader("Datos y limitaciones")

    for limitacion in metadata["limitaciones"]:
        st.write("- " + limitacion)


st.divider()

st.caption(
    "TechStore Analytics Integrador | MGR | "
    "Aplicación demostrativa basada en datos históricos"
)
