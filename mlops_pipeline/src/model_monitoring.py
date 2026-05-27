"""
model_monitoring.py
===================
Módulo de monitoreo y detección de data drift para el pipeline MLOps de predicción de riesgo crediticio.

Este módulo implementa:
- Cálculo de métricas de drift: KS test, PSI, Jensen-Shannon divergence, Chi-cuadrado
- Detección automática de drift por tipo de variable
- Sistema de alertas con umbrales configurables
- Generación de reportes de drift

Autor: Juan Carlos López S.
Fecha: Mayo 2026
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN DE UMBRALES
# ============================================================================

THRESHOLDS = {
    'psi': {
        'ok': 0.1,
        'warning': 0.25
    },
    'p_value': {
        'warning': 0.05,
        'critical': 0.01
    },
    'jensen_shannon': {
        'ok': 0.1,
        'warning': 0.3
    }
}


# ============================================================================
# FUNCIONES DE CÁLCULO DE DRIFT
# ============================================================================

def calculate_ks_test(data_reference, data_current, feature):
    """
    Calcula el test de Kolmogorov-Smirnov para detectar diferencias en distribuciones continuas.
    
    Parámetros:
    -----------
    data_reference : pd.DataFrame
        Datos de referencia (históricos)
    data_current : pd.DataFrame
        Datos actuales (producción)
    feature : str
        Nombre de la variable a analizar
    
    Retorna:
    --------
    dict : Diccionario con estadístico KS, p-value y estado de alerta
    """
    
    ref_values = data_reference[feature].dropna()
    curr_values = data_current[feature].dropna()
    
    # Aplicar test KS
    ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
    
    # Determinar estado de alerta
    if p_value > THRESHOLDS['p_value']['warning']:
        status = 'OK'
        alert = '🟢'
    elif p_value > THRESHOLDS['p_value']['critical']:
        status = 'WARNING'
        alert = '🟡'
    else:
        status = 'CRITICAL'
        alert = '🔴'
    
    return {
        'feature': feature,
        'metric': 'KS Test',
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'status': status,
        'alert': alert,
        'interpretation': f"p-value = {p_value:.4f} → {status}"
    }


def calculate_psi(data_reference, data_current, feature, bins=10):
    """
    Calcula el Population Stability Index (PSI) para medir estabilidad poblacional.
    
    PSI < 0.1: Sin cambio significativo
    0.1 <= PSI < 0.25: Cambio moderado
    PSI >= 0.25: Cambio significativo
    
    Parámetros:
    -----------
    data_reference : pd.DataFrame
        Datos de referencia
    data_current : pd.DataFrame
        Datos actuales
    feature : str
        Nombre de la variable
    bins : int
        Número de bins para discretizar (default: 10)
    
    Retorna:
    --------
    dict : Diccionario con valor PSI y estado
    """
    
    ref_values = data_reference[feature].dropna()
    curr_values = data_current[feature].dropna()
    
    # Crear bins basados en datos de referencia
    _, bin_edges = np.histogram(ref_values, bins=bins)
    
    # Calcular distribuciones
    ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_values, bins=bin_edges)
    
    # Convertir a proporciones (evitar división por cero)
    ref_props = (ref_counts + 1) / (ref_counts.sum() + bins)
    curr_props = (curr_counts + 1) / (curr_counts.sum() + bins)
    
    # Calcular PSI
    psi_value = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    
    # Determinar estado
    if psi_value < THRESHOLDS['psi']['ok']:
        status = 'OK'
        alert = '🟢'
    elif psi_value < THRESHOLDS['psi']['warning']:
        status = 'WARNING'
        alert = '🟡'
    else:
        status = 'CRITICAL'
        alert = '🔴'
    
    return {
        'feature': feature,
        'metric': 'PSI',
        'psi_value': psi_value,
        'status': status,
        'alert': alert,
        'interpretation': f"PSI = {psi_value:.4f} → {status}"
    }


def calculate_jensen_shannon(data_reference, data_current, feature, bins=10):
    """
    Calcula la divergencia de Jensen-Shannon entre dos distribuciones.
    
    Mide la similitud entre dos distribuciones de probabilidad.
    Valores: 0 (idénticas) a 1 (completamente diferentes)
    
    Parámetros:
    -----------
    data_reference : pd.DataFrame
        Datos de referencia
    data_current : pd.DataFrame
        Datos actuales
    feature : str
        Nombre de la variable
    bins : int
        Número de bins para discretizar
    
    Retorna:
    --------
    dict : Diccionario con distancia JS y estado
    """
    
    ref_values = data_reference[feature].dropna()
    curr_values = data_current[feature].dropna()
    
    # Crear bins
    _, bin_edges = np.histogram(ref_values, bins=bins)
    
    # Calcular distribuciones normalizadas
    ref_hist, _ = np.histogram(ref_values, bins=bin_edges, density=True)
    curr_hist, _ = np.histogram(curr_values, bins=bin_edges, density=True)
    
    # Normalizar para sumar 1 (distribución de probabilidad)
    ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
    curr_hist = curr_hist / curr_hist.sum() if curr_hist.sum() > 0 else curr_hist
    
    # Calcular divergencia Jensen-Shannon
    js_distance = jensenshannon(ref_hist, curr_hist)
    
    # Determinar estado
    if js_distance < THRESHOLDS['jensen_shannon']['ok']:
        status = 'OK'
        alert = '🟢'
    elif js_distance < THRESHOLDS['jensen_shannon']['warning']:
        status = 'WARNING'
        alert = '🟡'
    else:
        status = 'CRITICAL'
        alert = '🔴'
    
    return {
        'feature': feature,
        'metric': 'Jensen-Shannon',
        'js_distance': js_distance,
        'status': status,
        'alert': alert,
        'interpretation': f"JS = {js_distance:.4f} → {status}"
    }


def calculate_chi_square(data_reference, data_current, feature):
    """
    Calcula el test de Chi-cuadrado para variables categóricas.
    
    Parámetros:
    -----------
    data_reference : pd.DataFrame
        Datos de referencia
    data_current : pd.DataFrame
        Datos actuales
    feature : str
        Nombre de la variable categórica
    
    Retorna:
    --------
    dict : Diccionario con estadístico Chi2, p-value y estado
    """
    
    ref_values = data_reference[feature].dropna()
    curr_values = data_current[feature].dropna()
    
    # Obtener todas las categorías únicas
    all_categories = set(ref_values.unique()) | set(curr_values.unique())
    
    # Crear tabla de contingencia
    ref_counts = ref_values.value_counts().reindex(all_categories, fill_value=0)
    curr_counts = curr_values.value_counts().reindex(all_categories, fill_value=0)
    
    # Crear matriz de contingencia
    contingency_table = np.array([ref_counts.values, curr_counts.values])
    
    # Aplicar test Chi-cuadrado
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Determinar estado
    if p_value > THRESHOLDS['p_value']['warning']:
        status = 'OK'
        alert = '🟢'
    elif p_value > THRESHOLDS['p_value']['critical']:
        status = 'WARNING'
        alert = '🟡'
    else:
        status = 'CRITICAL'
        alert = '🔴'
    
    return {
        'feature': feature,
        'metric': 'Chi-Square',
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'dof': dof,
        'status': status,
        'alert': alert,
        'interpretation': f"p-value = {p_value:.4f} → {status}"
    }


# ============================================================================
# DETECCIÓN AUTOMÁTICA DE DRIFT
# ============================================================================

def detect_drift_all_features(data_reference, data_current, numeric_features, categorical_features):
    """
    Detecta drift en todas las features aplicando las métricas apropiadas según tipo de variable.
    
    Parámetros:
    -----------
    data_reference : pd.DataFrame
        Datos de referencia (históricos)
    data_current : pd.DataFrame
        Datos actuales (producción)
    numeric_features : list
        Lista de features numéricas
    categorical_features : list
        Lista de features categóricas
    
    Retorna:
    --------
    pd.DataFrame : DataFrame con resultados de drift por feature
    """
    
    results = []
    
    print(f"\n{'='*80}")
    print("INICIANDO DETECCIÓN DE DATA DRIFT")
    print(f"{'='*80}")
    print(f"\n✓ Datos de referencia: {len(data_reference)} registros")
    print(f"✓ Datos actuales: {len(data_current)} registros")
    print(f"✓ Features numéricas: {len(numeric_features)}")
    print(f"✓ Features categóricas: {len(categorical_features)}")
    
    # Analizar features numéricas
    print(f"\n{'='*80}")
    print("ANALIZANDO FEATURES NUMÉRICAS")
    print(f"{'='*80}")
    
    for feature in numeric_features:
        if feature in data_reference.columns and feature in data_current.columns:
            print(f"\n  Analizando: {feature}")
            
            # Calcular múltiples métricas
            ks_result = calculate_ks_test(data_reference, data_current, feature)
            psi_result = calculate_psi(data_reference, data_current, feature)
            js_result = calculate_jensen_shannon(data_reference, data_current, feature)
            
            # Consolidar resultados
            result = {
                'feature': feature,
                'type': 'numeric',
                'ks_statistic': ks_result['ks_statistic'],
                'ks_p_value': ks_result['p_value'],
                'psi': psi_result['psi_value'],
                'jensen_shannon': js_result['js_distance'],
                'status': _determine_overall_status([ks_result, psi_result, js_result]),
                'alert': _determine_overall_alert([ks_result, psi_result, js_result])
            }
            
            results.append(result)
            print(f"    • KS: {ks_result['interpretation']}")
            print(f"    • PSI: {psi_result['interpretation']}")
            print(f"    • JS: {js_result['interpretation']}")
            print(f"    → Estado final: {result['alert']} {result['status']}")
    
    # Analizar features categóricas
    print(f"\n{'='*80}")
    print("ANALIZANDO FEATURES CATEGÓRICAS")
    print(f"{'='*80}")
    
    for feature in categorical_features:
        if feature in data_reference.columns and feature in data_current.columns:
            print(f"\n  Analizando: {feature}")
            
            # Calcular Chi-cuadrado
            chi2_result = calculate_chi_square(data_reference, data_current, feature)
            
            result = {
                'feature': feature,
                'type': 'categorical',
                'chi2_statistic': chi2_result['chi2_statistic'],
                'chi2_p_value': chi2_result['p_value'],
                'dof': chi2_result['dof'],
                'psi': None,
                'jensen_shannon': None,
                'status': chi2_result['status'],
                'alert': chi2_result['alert']
            }
            
            results.append(result)
            print(f"    • Chi2: {chi2_result['interpretation']}")
            print(f"    → Estado final: {result['alert']} {result['status']}")
    
    # Crear DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    return df_results


def _determine_overall_status(metric_results):
    """
    Determina el estado general basado en múltiples métricas.
    Si alguna métrica está en CRITICAL, el estado general es CRITICAL.
    Si alguna está en WARNING, el estado es WARNING.
    Si todas están OK, el estado es OK.
    """
    statuses = [r['status'] for r in metric_results]
    
    if 'CRITICAL' in statuses:
        return 'CRITICAL'
    elif 'WARNING' in statuses:
        return 'WARNING'
    else:
        return 'OK'


def _determine_overall_alert(metric_results):
    """Determina el emoji de alerta general."""
    status = _determine_overall_status(metric_results)
    
    if status == 'CRITICAL':
        return '🔴'
    elif status == 'WARNING':
        return '🟡'
    else:
        return '🟢'


# ============================================================================
# GENERACIÓN DE REPORTES
# ============================================================================

def generate_drift_report(drift_results):
    """
    Genera un reporte textual resumido de drift.
    
    Parámetros:
    -----------
    drift_results : pd.DataFrame
        Resultados de detección de drift
    
    Retorna:
    --------
    str : Reporte textual con resumen y recomendaciones
    """
    
    print(f"\n{'='*80}")
    print("REPORTE DE DATA DRIFT")
    print(f"{'='*80}")
    
    # Contar por estado
    total_features = len(drift_results)
    critical_count = len(drift_results[drift_results['status'] == 'CRITICAL'])
    warning_count = len(drift_results[drift_results['status'] == 'WARNING'])
    ok_count = len(drift_results[drift_results['status'] == 'OK'])
    
    print(f"\n📊 Resumen General:")
    print(f"   • Total de features analizadas: {total_features}")
    print(f"   • 🟢 OK: {ok_count} ({ok_count/total_features*100:.1f}%)")
    print(f"   • 🟡 WARNING: {warning_count} ({warning_count/total_features*100:.1f}%)")
    print(f"   • 🔴 CRITICAL: {critical_count} ({critical_count/total_features*100:.1f}%)")
    
    # Features críticas
    if critical_count > 0:
        print(f"\n⚠️ Features con DRIFT CRÍTICO:")
        critical_features = drift_results[drift_results['status'] == 'CRITICAL']
        for idx, row in critical_features.iterrows():
            print(f"   • {row['feature']} ({row['type']})")
            if row['type'] == 'numeric':
                print(f"     - PSI: {row['psi']:.4f}")
                print(f"     - KS p-value: {row['ks_p_value']:.4f}")
            else:
                print(f"     - Chi2 p-value: {row['chi2_p_value']:.4f}")
    
    # Features con warning
    if warning_count > 0:
        print(f"\n⚡ Features con DRIFT MODERADO:")
        warning_features = drift_results[drift_results['status'] == 'WARNING']
        for idx, row in warning_features.iterrows():
            print(f"   • {row['feature']} ({row['type']})")
    
    # Recomendaciones
    print(f"\n💡 Recomendaciones:")
    
    if critical_count > 0:
        print(f"   🔴 ACCIÓN INMEDIATA REQUERIDA:")
        print(f"      - {critical_count} features presentan drift significativo")
        print(f"      - Considerar reentrenamiento del modelo")
        print(f"      - Revisar cambios en el proceso de recolección de datos")
        print(f"      - Analizar si las features críticas son importantes para el modelo")
    
    if warning_count > 0:
        print(f"   🟡 MONITOREO CONTINUO:")
        print(f"      - {warning_count} features muestran cambios moderados")
        print(f"      - Establecer plan de seguimiento semanal")
        print(f"      - Preparar estrategia de reentrenamiento preventivo")
    
    if ok_count == total_features:
        print(f"   🟢 SISTEMA ESTABLE:")
        print(f"      - Todas las features están dentro de los umbrales esperados")
        print(f"      - Continuar con monitoreo regular")
    
    print(f"\n{'='*80}")
    
    return drift_results


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal para ejecutar el monitoreo de drift.
    """
    
    # Importar módulo de carga de datos
    from cargar_datos import load_data
    
    print("="*80)
    print("SISTEMA DE MONITOREO DE DATA DRIFT")
    print("="*80)
    
    # Cargar datos completos
    print("\n1. Cargando datos...")
    df = load_data('../../Base_de_datos.xlsx')
    
    # Simular división: 70% referencia (históricos) vs 30% actual (producción)
    print("\n2. Dividiendo datos en referencia vs actual...")
    split_point = int(len(df) * 0.7)
    
    data_reference = df.iloc[:split_point].copy()
    data_current = df.iloc[split_point:].copy()
    
    print(f"   ✓ Datos de referencia: {len(data_reference)} registros (primeros 70%)")
    print(f"   ✓ Datos actuales: {len(data_current)} registros (últimos 30%)")
    
    # Eliminar target para análisis de drift
    target_col = 'Pago_atiempo'
    if target_col in data_reference.columns:
        data_reference = data_reference.drop(target_col, axis=1)
        data_current = data_current.drop(target_col, axis=1)
    
    # Identificar tipos de variables
    print("\n3. Identificando tipos de variables...")
    
    numeric_features = data_reference.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data_reference.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Excluir fecha_prestamo si existe (es temporal, no para drift en este contexto)
    if 'fecha_prestamo' in numeric_features:
        numeric_features.remove('fecha_prestamo')
    if 'fecha_prestamo' in categorical_features:
        categorical_features.remove('fecha_prestamo')
    
    print(f"   ✓ Features numéricas detectadas: {len(numeric_features)}")
    print(f"   ✓ Features categóricas detectadas: {len(categorical_features)}")
    
    # Ejecutar detección de drift
    print("\n4. Ejecutando detección de drift...")
    drift_results = detect_drift_all_features(
        data_reference, 
        data_current, 
        numeric_features, 
        categorical_features
    )
    
    # Generar reporte
    print("\n5. Generando reporte...")
    generate_drift_report(drift_results)
    
    # Mostrar tabla de resultados
    print(f"\n{'='*80}")
    print("TABLA DE RESULTADOS DETALLADOS")
    print(f"{'='*80}")
    print(drift_results.to_string(index=False))
    print(f"{'='*80}")
    
    print(f"\n✓ MONITOREO DE DRIFT COMPLETADO EXITOSAMENTE")
    print(f"{'='*80}\n")
    
    return drift_results


if __name__ == "__main__":
    results = main()
