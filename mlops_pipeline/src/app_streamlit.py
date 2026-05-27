"""
Dashboard interactivo para monitoreo de Data Drift
Proyecto MLOps - Avance #3
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importar funciones del módulo de monitoreo
from model_monitoring import (
    calculate_ks_test,
    calculate_psi,
    calculate_jensen_shannon,
    calculate_chi_square,
    detect_drift_all_features
)
from cargar_datos import load_data

# ================================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ================================================================================

st.set_page_config(
    page_title="MLOps - Monitoreo de Drift",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# ESTILOS PERSONALIZADOS
# ================================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .alert-warning {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
    }
    .alert-ok {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# FUNCIONES AUXILIARES
# ================================================================================

@st.cache_data
def load_and_analyze_data(file_path, reference_ratio=0.7):
    """
    Carga datos y ejecuta análisis de drift
    """
    # Cargar datos
    df = load_data(file_path)
    
    # Dividir en referencia y actual
    split_index = int(len(df) * reference_ratio)
    data_reference = df.iloc[:split_index].copy()
    data_current = df.iloc[split_index:].copy()
    
    # Identificar tipos de features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir la columna objetivo si existe
    if 'Pago_atiempo' in numeric_features:
        numeric_features.remove('Pago_atiempo')
    
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Ejecutar detección de drift
    drift_results = detect_drift_all_features(
        data_reference, 
        data_current, 
        numeric_features, 
        categorical_features
    )
    
    return df, data_reference, data_current, drift_results, numeric_features, categorical_features


def get_status_color(status):
    """Retorna color según el estado"""
    colors = {
        'OK': '#4caf50',
        'WARNING': '#ff9800',
        'CRITICAL': '#f44336'
    }
    return colors.get(status, '#9e9e9e')


def create_distribution_comparison(data_ref, data_curr, feature, feature_type='numeric'):
    """
    Crea gráfico comparativo de distribuciones
    """
    if feature_type == 'numeric':
        # Histogramas superpuestos
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data_ref[feature],
            name='Referencia',
            opacity=0.7,
            marker_color='#1f77b4',
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=data_curr[feature],
            name='Actual',
            opacity=0.7,
            marker_color='#ff7f0e',
            nbinsx=30
        ))
        
        fig.update_layout(
            title=f'Distribución: {feature}',
            xaxis_title=feature,
            yaxis_title='Frecuencia',
            barmode='overlay',
            height=400,
            showlegend=True
        )
        
    else:
        # Gráfico de barras para categóricas
        ref_counts = data_ref[feature].value_counts()
        curr_counts = data_curr[feature].value_counts()
        
        categories = sorted(set(ref_counts.index) | set(curr_counts.index))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=[ref_counts.get(cat, 0) for cat in categories],
            name='Referencia',
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            x=categories,
            y=[curr_counts.get(cat, 0) for cat in categories],
            name='Actual',
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title=f'Distribución: {feature}',
            xaxis_title=feature,
            yaxis_title='Frecuencia',
            barmode='group',
            height=400,
            showlegend=True
        )
    
    return fig


def create_metrics_radar(drift_results):
    """
    Crea gráfico de radar con métricas de drift
    """
    # Seleccionar solo features numéricas con todas las métricas
    numeric_drift = drift_results[drift_results['type'] == 'numeric'].copy()
    
    # Tomar las 10 features con mayor drift (según PSI)
    top_drift = numeric_drift.nlargest(10, 'psi')
    
    fig = go.Figure()
    
    # PSI normalizado
    psi_normalized = np.clip(top_drift['psi'].values / 0.3, 0, 1)
    
    # KS statistic
    ks_values = top_drift['ks_statistic'].values
    
    # JS distance
    js_values = top_drift['jensen_shannon'].values
    
    fig.add_trace(go.Scatterpolar(
        r=psi_normalized,
        theta=top_drift['feature'].values,
        fill='toself',
        name='PSI (normalizado)',
        line_color='#1f77b4'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=ks_values,
        theta=top_drift['feature'].values,
        fill='toself',
        name='KS Statistic',
        line_color='#ff7f0e'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=js_values,
        theta=top_drift['feature'].values,
        fill='toself',
        name='Jensen-Shannon',
        line_color='#2ca02c'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='Top 10 Features con Mayor Drift (Múltiples Métricas)',
        height=500
    )
    
    return fig


def create_psi_comparison(drift_results):
    """
    Crea gráfico de barras horizontal con valores PSI
    """
    numeric_drift = drift_results[drift_results['type'] == 'numeric'].copy()
    numeric_drift = numeric_drift.sort_values('psi', ascending=True)
    
    # Colores según umbral
    colors = []
    for psi in numeric_drift['psi']:
        if psi < 0.1:
            colors.append('#4caf50')  # OK
        elif psi < 0.25:
            colors.append('#ff9800')  # WARNING
        else:
            colors.append('#f44336')  # CRITICAL
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=numeric_drift['psi'],
        y=numeric_drift['feature'],
        orientation='h',
        marker_color=colors,
        text=numeric_drift['psi'].round(3),
        textposition='auto'
    ))
    
    # Líneas de umbral
    fig.add_vline(x=0.1, line_dash="dash", line_color="green", 
                  annotation_text="Umbral OK (0.1)")
    fig.add_vline(x=0.25, line_dash="dash", line_color="orange", 
                  annotation_text="Umbral WARNING (0.25)")
    
    fig.update_layout(
        title='Population Stability Index (PSI) por Feature',
        xaxis_title='PSI Value',
        yaxis_title='Feature',
        height=600,
        showlegend=False
    )
    
    return fig


def create_status_summary_chart(drift_results):
    """
    Crea gráfico de resumen por estado
    """
    status_counts = drift_results['status'].value_counts()
    
    colors_map = {
        'OK': '#4caf50',
        'WARNING': '#ff9800',
        'CRITICAL': '#f44336'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=.4,
        marker_colors=[colors_map.get(status, '#9e9e9e') for status in status_counts.index],
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title='Distribución de Estados de Drift',
        height=400,
        showlegend=True
    )
    
    return fig


# ================================================================================
# SIDEBAR
# ================================================================================

st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("---")

# Ruta del archivo
file_path = st.sidebar.text_input(
    "Ruta del archivo de datos:",
    value="../../Base_de_datos.xlsx"
)

# Ratio de división
reference_ratio = st.sidebar.slider(
    "% de datos de referencia:",
    min_value=0.5,
    max_value=0.9,
    value=0.7,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 Información
**Proyecto:** MLOps Pipeline  
**Versión:** V1.2.0  
**Avance:** #3 - Monitoreo de Drift  

**Métricas implementadas:**
- 🔹 KS Test (Kolmogorov-Smirnov)
- 🔹 PSI (Population Stability Index)
- 🔹 Jensen-Shannon Distance
- 🔹 Chi-cuadrado (categóricas)
""")

# ================================================================================
# CARGA DE DATOS Y ANÁLISIS
# ================================================================================

st.markdown('<h1 class="main-header">📊 Dashboard de Monitoreo de Data Drift</h1>', 
            unsafe_allow_html=True)

try:
    with st.spinner('🔄 Cargando datos y ejecutando análisis de drift...'):
        df, data_reference, data_current, drift_results, numeric_features, categorical_features = load_and_analyze_data(
            file_path, 
            reference_ratio
        )
    
    st.success('✅ Análisis completado exitosamente')
    
    # ================================================================================
    # RESUMEN EJECUTIVO
    # ================================================================================
    
    st.markdown("## 📈 Resumen Ejecutivo")
    
    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_features = len(drift_results)
    critical_count = len(drift_results[drift_results['status'] == 'CRITICAL'])
    warning_count = len(drift_results[drift_results['status'] == 'WARNING'])
    ok_count = len(drift_results[drift_results['status'] == 'OK'])
    
    with col1:
        st.metric(
            label="📊 Total Features",
            value=total_features
        )
    
    with col2:
        st.metric(
            label="🔴 Críticas",
            value=critical_count,
            delta=f"{(critical_count/total_features)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="🟡 Advertencias",
            value=warning_count,
            delta=f"{(warning_count/total_features)*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="🟢 Estables",
            value=ok_count,
            delta=f"{(ok_count/total_features)*100:.1f}%"
        )
    
    with col5:
        st.metric(
            label="📦 Total Registros",
            value=f"{len(df):,}"
        )
    
    st.markdown("---")
    
    # ================================================================================
    # ALERTAS Y RECOMENDACIONES
    # ================================================================================
    
    if critical_count > 0:
        st.markdown('<div class="alert-critical">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🔴 ALERTA CRÍTICA
        
        **{critical_count} features presentan drift significativo ({(critical_count/total_features)*100:.1f}% del total)**
        
        **Acciones recomendadas:**
        - ⚠️ Revisar inmediatamente las features críticas
        - 🔄 Considerar reentrenamiento del modelo
        - 📊 Analizar cambios en el proceso de recolección de datos
        - 🎯 Evaluar el impacto en las predicciones actuales
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif warning_count > 0:
        st.markdown('<div class="alert-warning">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🟡 ADVERTENCIA
        
        **{warning_count} features presentan señales de drift ({(warning_count/total_features)*100:.1f}% del total)**
        
        **Acciones recomendadas:**
        - 👀 Monitorear de cerca estas features
        - 📈 Analizar tendencias temporales
        - 🔍 Verificar calidad de datos entrantes
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="alert-ok">', unsafe_allow_html=True)
        st.markdown("""
        ### 🟢 ESTADO ÓPTIMO
        
        **No se detectó drift significativo en ninguna feature**
        
        El modelo se encuentra operando en condiciones estables.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================================
    # TABLA INTERACTIVA DE RESULTADOS
    # ================================================================================
    
    st.markdown("## 📋 Resultados Detallados por Feature")
    
    # Formatear la tabla para mejor visualización
    display_df = drift_results.copy()
    
    # Redondear valores numéricos
    numeric_columns = ['ks_statistic', 'ks_p_value', 'psi', 'jensen_shannon', 
                       'chi2_statistic', 'chi2_p_value']
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)
    
    # Aplicar estilo condicional
    def highlight_status(row):
        if row['status'] == 'CRITICAL':
            return ['background-color: #ffebee'] * len(row)
        elif row['status'] == 'WARNING':
            return ['background-color: #fff3e0'] * len(row)
        else:
            return ['background-color: #e8f5e9'] * len(row)
    
    styled_df = display_df.style.apply(highlight_status, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Botón de descarga
    csv = drift_results.to_csv(index=False)
    st.download_button(
        label="📥 Descargar resultados (CSV)",
        data=csv,
        file_name="drift_results.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # ================================================================================
    # VISUALIZACIONES
    # ================================================================================
    
    st.markdown("## 📊 Visualizaciones")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Resumen General",
        "📈 Comparación de Distribuciones",
        "📊 Métricas de Drift",
        "🔍 Análisis por Feature"
    ])
    
    with tab1:
        st.markdown("### Resumen General de Drift")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de pastel
            fig_pie = create_status_summary_chart(drift_results)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gráfico de barras por tipo
            type_status = drift_results.groupby(['type', 'status']).size().reset_index(name='count')
            fig_bar = px.bar(
                type_status,
                x='type',
                y='count',
                color='status',
                title='Distribución de Drift por Tipo de Feature',
                color_discrete_map={
                    'OK': '#4caf50',
                    'WARNING': '#ff9800',
                    'CRITICAL': '#f44336'
                },
                barmode='group'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.markdown("### Comparación de Distribuciones")
        
        # Selector de feature
        all_features = numeric_features + categorical_features
        selected_feature = st.selectbox(
            "Selecciona una feature para visualizar:",
            options=all_features,
            index=0 if all_features else None
        )
        
        if selected_feature:
            feature_type = 'numeric' if selected_feature in numeric_features else 'categorical'
            
            # Obtener estado de drift de esta feature
            feature_status = drift_results[drift_results['feature'] == selected_feature]['status'].values[0]
            feature_alert = drift_results[drift_results['feature'] == selected_feature]['alert'].values[0]
            
            st.markdown(f"**Estado:** {feature_alert} {feature_status}")
            
            # Gráfico de distribución
            fig_dist = create_distribution_comparison(
                data_reference, 
                data_current, 
                selected_feature, 
                feature_type
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Mostrar métricas específicas
            feature_metrics = drift_results[drift_results['feature'] == selected_feature].iloc[0]
            
            st.markdown("#### Métricas de Drift")
            
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                if not pd.isna(feature_metrics['ks_statistic']):
                    st.metric("KS Statistic", f"{feature_metrics['ks_statistic']:.4f}")
                    st.metric("KS p-value", f"{feature_metrics['ks_p_value']:.4e}")
            
            with metric_cols[1]:
                if not pd.isna(feature_metrics['psi']):
                    st.metric("PSI", f"{feature_metrics['psi']:.4f}")
            
            with metric_cols[2]:
                if not pd.isna(feature_metrics['jensen_shannon']):
                    st.metric("Jensen-Shannon", f"{feature_metrics['jensen_shannon']:.4f}")
            
            with metric_cols[3]:
                if not pd.isna(feature_metrics['chi2_statistic']):
                    st.metric("Chi² Statistic", f"{feature_metrics['chi2_statistic']:.4f}")
                    st.metric("Chi² p-value", f"{feature_metrics['chi2_p_value']:.4e}")
    
    with tab3:
        st.markdown("### Métricas de Drift")
        
        # PSI comparison
        st.markdown("#### Population Stability Index (PSI)")
        fig_psi = create_psi_comparison(drift_results)
        st.plotly_chart(fig_psi, use_container_width=True)
        
        st.markdown("---")
        
        # Radar chart
        st.markdown("#### Comparación Multidimensional (Top 10 Features)")
        fig_radar = create_metrics_radar(drift_results)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab4:
        st.markdown("### Análisis Detallado por Feature")
        
        # Filtros
        col1, col2 = st.columns(2)
        
        with col1:
            filter_type = st.multiselect(
                "Filtrar por tipo:",
                options=['numeric', 'categorical'],
                default=['numeric', 'categorical']
            )
        
        with col2:
            filter_status = st.multiselect(
                "Filtrar por estado:",
                options=['OK', 'WARNING', 'CRITICAL'],
                default=['CRITICAL', 'WARNING', 'OK']
            )
        
        # Aplicar filtros
        filtered_results = drift_results[
            (drift_results['type'].isin(filter_type)) &
            (drift_results['status'].isin(filter_status))
        ]
        
        st.markdown(f"**Features filtradas:** {len(filtered_results)}")
        
        # Tabla filtrada
        st.dataframe(filtered_results, use_container_width=True, height=400)
        
        # Gráfico de dispersión: PSI vs KS
        if len(filtered_results[filtered_results['type'] == 'numeric']) > 0:
            numeric_filtered = filtered_results[filtered_results['type'] == 'numeric']
            
            fig_scatter = px.scatter(
                numeric_filtered,
                x='ks_statistic',
                y='psi',
                color='status',
                size='jensen_shannon',
                hover_data=['feature'],
                title='Relación entre KS Statistic y PSI',
                color_discrete_map={
                    'OK': '#4caf50',
                    'WARNING': '#ff9800',
                    'CRITICAL': '#f44336'
                },
                labels={
                    'ks_statistic': 'KS Statistic',
                    'psi': 'Population Stability Index',
                    'jensen_shannon': 'Jensen-Shannon Distance'
                }
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================================================
    # INFORMACIÓN TÉCNICA
    # ================================================================================
    
    with st.expander("ℹ️ Información Técnica sobre las Métricas"):
        st.markdown("""
        ### 📚 Métricas de Data Drift Implementadas
        
        #### 1. **KS Test (Kolmogorov-Smirnov)**
        - **Aplicación:** Variables numéricas
        - **Descripción:** Prueba estadística que mide la distancia máxima entre las funciones de distribución acumulada
        - **Interpretación p-value:**
          - p > 0.05: No hay evidencia de drift (OK)
          - 0.01 < p ≤ 0.05: Posible drift (WARNING)
          - p ≤ 0.01: Drift significativo (CRITICAL)
        
        #### 2. **PSI (Population Stability Index)**
        - **Aplicación:** Variables numéricas
        - **Descripción:** Mide el cambio en la distribución comparando proporciones entre bins
        - **Umbrales:**
          - PSI < 0.1: No hay cambio significativo (OK)
          - 0.1 ≤ PSI < 0.25: Cambio moderado (WARNING)
          - PSI ≥ 0.25: Cambio significativo (CRITICAL)
        - **Fórmula:** PSI = Σ (% actual - % referencia) × ln(% actual / % referencia)
        
        #### 3. **Jensen-Shannon Distance**
        - **Aplicación:** Variables numéricas
        - **Descripción:** Medida de similitud entre distribuciones de probabilidad (0 = idénticas, 1 = completamente diferentes)
        - **Umbrales:**
          - JS < 0.1: Distribuciones similares (OK)
          - 0.1 ≤ JS < 0.3: Diferencia moderada (WARNING)
          - JS ≥ 0.3: Diferencia significativa (CRITICAL)
        
        #### 4. **Chi-cuadrado (χ²)**
        - **Aplicación:** Variables categóricas
        - **Descripción:** Prueba de independencia que evalúa si hay cambios en las frecuencias de categorías
        - **Interpretación p-value:**
          - p > 0.05: No hay evidencia de drift (OK)
          - 0.01 < p ≤ 0.05: Posible drift (WARNING)
          - p ≤ 0.01: Drift significativo (CRITICAL)
        
        ---
        
        ### 🎯 Recomendaciones de Uso
        
        - **Monitoreo Continuo:** Ejecutar este análisis regularmente (semanal/mensual)
        - **Múltiples Métricas:** Usar las 3 métricas para variables numéricas da mayor confianza en la detección
        - **Contexto de Negocio:** Interpretar los resultados considerando el contexto del negocio
        - **Acción Temprana:** Actuar ante señales WARNING antes de que se vuelvan CRITICAL
        - **Reentrenamiento:** Si >30% de features están en CRITICAL, considerar reentrenamiento del modelo
        """)

except Exception as e:
    st.error(f"❌ Error al cargar o analizar los datos: {str(e)}")
    st.info("Verifica que la ruta del archivo sea correcta y que el archivo exista.")
    st.exception(e)

# ================================================================================
# FOOTER
# ================================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>MLOps Pipeline - Data Drift Monitoring System</strong></p>
    <p>Proyecto Integrador M5 | Versión 1.2.0 | Avance #3</p>
</div>
""", unsafe_allow_html=True)
