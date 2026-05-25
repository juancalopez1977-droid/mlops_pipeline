"""
Módulo para carga y validación de datos del proyecto MLOps - Riesgo de Crédito
Autor: Juan Carlos López S.
Fecha: Mayo 2026
Descripción: Carga y valida el dataset Base_de_datos.xlsx con información histórica de créditos
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga el archivo Excel de datos históricos de crédito.
    
    Parameters
    ----------
    file_path : str
        Ruta al archivo Base_de_datos.xlsx
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados
        
    Raises
    ------
    FileNotFoundError
        Si el archivo no existe
    ValueError
        Si el archivo está vacío o corrupto
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"✓ Datos cargados exitosamente: {df.shape[0]} registros x {df.shape[1]} columnas")
        return df
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {str(e)}")


def validate_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
    """
    Valida la integridad y estructura del DataFrame cargado.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar
        
    Returns
    -------
    Tuple[bool, Dict]
        (is_valid, validation_report) donde:
        - is_valid: True si pasa todas las validaciones
        - validation_report: diccionario con métricas de validación
    """
    validation_report = {}
    is_valid = True
    
    # Validación 1: Columnas esperadas
    expected_columns = [
        'tipo_credito', 'fecha_prestamo', 'capital_prestado', 'plazo_meses',
        'edad_cliente', 'tipo_laboral', 'salario_cliente', 'total_otros_prestamos',
        'cuota_pactada', 'puntaje', 'puntaje_datacredito', 'cant_creditosvigentes',
        'huella_consulta', 'saldo_mora', 'saldo_total', 'saldo_principal',
        'saldo_mora_codeudor', 'creditos_sectorFinanciero', 'creditos_sectorCooperativo',
        'creditos_sectorReal', 'promedio_ingresos_datacredito', 'tendencia_ingresos',
        'Pago_atiempo'
    ]
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        is_valid = False
        validation_report['columnas_faltantes'] = list(missing_cols)
    else:
        validation_report['columnas_faltantes'] = []
    
    # Validación 2: Variable objetivo presente
    if 'Pago_atiempo' not in df.columns:
        is_valid = False
        validation_report['target_presente'] = False
    else:
        validation_report['target_presente'] = True
        validation_report['distribucion_target'] = df['Pago_atiempo'].value_counts().to_dict()
        
        # Calcular balance de clases
        target_counts = df['Pago_atiempo'].value_counts()
        if len(target_counts) == 2:
            balance_ratio = target_counts.min() / target_counts.max()
            validation_report['balance_clases'] = round(balance_ratio, 3)
            if balance_ratio < 0.2:
                print(f"⚠ Advertencia: Desbalance de clases detectado (ratio: {balance_ratio:.3f})")
    
    # Validación 3: Tipos de datos
    validation_report['tipos_datos'] = df.dtypes.astype(str).to_dict()
    
    # Validación 4: Valores nulos
    null_counts = df.isnull().sum()
    validation_report['valores_nulos'] = null_counts[null_counts > 0].to_dict()
    validation_report['porcentaje_nulos_total'] = round((df.isnull().sum().sum() / df.size) * 100, 2)
    
    # Validación 5: Duplicados
    duplicates = df.duplicated().sum()
    validation_report['registros_duplicados'] = int(duplicates)
    if duplicates > 0:
        print(f"⚠ Se encontraron {duplicates} registros duplicados")
    
    # Validación 6: Dimensiones
    validation_report['dimensiones'] = {
        'filas': df.shape[0],
        'columnas': df.shape[1]
    }
    
    if df.shape[0] < 100:
        is_valid = False
        print("✗ Error: Dataset muy pequeño para modelado (<100 registros)")
    
    # Validación 7: Valores fuera de rango lógicos
    validation_issues = []
    
    if 'edad_cliente' in df.columns:
        if (df['edad_cliente'] < 18).any() or (df['edad_cliente'] > 100).any():
            validation_issues.append('edad_cliente fuera de rango [18, 100]')
    
    if 'salario_cliente' in df.columns:
        if (df['salario_cliente'] < 0).any():
            validation_issues.append('salario_cliente con valores negativos')
    
    if 'capital_prestado' in df.columns:
        if (df['capital_prestado'] <= 0).any():
            validation_issues.append('capital_prestado con valores <= 0')
    
    validation_report['valores_fuera_rango'] = validation_issues
    if validation_issues:
        print(f"⚠ Advertencia: {len(validation_issues)} tipos de valores fuera de rango detectados")
    
    # Resumen final
    if is_valid:
        print("✓ Validación exitosa: Dataset cumple con los requisitos mínimos")
    else:
        print("✗ Validación fallida: Revisar reporte de validación")
    
    return is_valid, validation_report


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Imprime un resumen estadístico del DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a resumir
    """
    print("\n" + "="*80)
    print("RESUMEN DE DATOS")
    print("="*80)
    
    print(f"\nDimensiones: {df.shape[0]} registros x {df.shape[1]} columnas")
    
    print(f"\nVariables numéricas: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"Variables categóricas: {df.select_dtypes(include=['object']).shape[1]}")
    print(f"Variables fecha: {df.select_dtypes(include=['datetime64']).shape[1]}")
    
    print("\nValores nulos por columna:")
    null_summary = df.isnull().sum()
    null_summary = null_summary[null_summary > 0].sort_values(ascending=False)
    if len(null_summary) > 0:
        for col, count in null_summary.items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")
    else:
        print("  ✓ No hay valores nulos")
    
    if 'Pago_atiempo' in df.columns:
        print("\nDistribución variable objetivo (Pago_atiempo):")
        target_dist = df['Pago_atiempo'].value_counts()
        for val, count in target_dist.items():
            pct = (count / len(df)) * 100
            label = "Pagó a tiempo" if val == 1 else "No pagó a tiempo"
            print(f"  - {label} ({val}): {count} ({pct:.2f}%)")
    
    print("="*80)


if __name__ == "__main__":
    # Ejecución de prueba del módulo
    print("Iniciando carga y validación de datos...\n")
    
    # Cargar datos
    df = load_data('../../Base_de_datos.xlsx')
    
    # Validar datos
    is_valid, report = validate_data(df)
    
    # Mostrar resumen
    get_data_summary(df)
    
    # Guardar reporte de validación
    print("\n✓ Proceso completado. DataFrame listo para análisis exploratorio.")