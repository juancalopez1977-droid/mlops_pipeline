"""
ft_engineering.py
=================
Módulo de Feature Engineering para el pipeline MLOps de predicción de riesgo crediticio.

Este módulo implementa las transformaciones identificadas en el EDA:
- Imputación de valores nulos
- Creación de features de interacción
- Encoding de variables categóricas
- Extracción de features temporales
- Normalización/estandarización
- Pipeline completo de transformación

Autor: Juan Carlos López S.
Fecha: Mayo 2026
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TRANSFORMADORES PERSONALIZADOS
# ============================================================================

class InteractionFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Crea features de interacción relevantes para riesgo crediticio.
    
    Features creadas:
    - ratio_cuota_salario: Capacidad de pago
    - ratio_capital_salario: Tamaño del préstamo relativo
    - ratio_otros_prestamos_salario: Carga de deuda existente
    - carga_deuda_total: Endeudamiento total
    - edad_fin_credito: Edad al terminar el préstamo
    - total_creditos_activos: Suma de créditos en todos los sectores
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Ratio cuota / salario (capacidad de pago)
        X['ratio_cuota_salario'] = X['cuota_pactada'] / (X['salario_cliente'] + 1)
        
        # Ratio capital / salario
        X['ratio_capital_salario'] = X['capital_prestado'] / (X['salario_cliente'] + 1)
        
        # Ratio otros préstamos / salario
        X['ratio_otros_prestamos_salario'] = X['total_otros_prestamos'] / (X['salario_cliente'] + 1)
        
        # Carga total de deuda
        X['carga_deuda_total'] = (X['cuota_pactada'] + X['total_otros_prestamos']) / (X['salario_cliente'] + 1)
        
        # Edad al finalizar el crédito
        X['edad_fin_credito'] = X['edad_cliente'] + (X['plazo_meses'] / 12)
        
        # Total créditos activos (suma de sectores)
        X['total_creditos_activos'] = (X['creditos_sectorFinanciero'] + 
                                        X['creditos_sectorCooperativo'] + 
                                        X['creditos_sectorReal'])
        
        return X


class TemporalFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Extrae features temporales de la variable fecha_prestamo.
    
    Features creadas:
    - año_prestamo
    - mes_prestamo
    - dia_semana_prestamo
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'fecha_prestamo' in X.columns:
            # Convertir a datetime si no lo es
            if not pd.api.types.is_datetime64_any_dtype(X['fecha_prestamo']):
                X['fecha_prestamo'] = pd.to_datetime(X['fecha_prestamo'])
            
            # Extraer componentes temporales
            X['año_prestamo'] = X['fecha_prestamo'].dt.year
            X['mes_prestamo'] = X['fecha_prestamo'].dt.month
            X['dia_semana_prestamo'] = X['fecha_prestamo'].dt.dayofweek
            
            # Eliminar la columna original
            X = X.drop('fecha_prestamo', axis=1)
        
        return X


class DatacreditoFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Crea feature binaria indicando si el cliente tiene historial de Datacredito.
    
    Feature creada:
    - sin_historial_datacredito: 1 si faltan datos de Datacredito, 0 si no
    """
    
    def __init__(self, datacredito_cols=None):
        self.datacredito_cols = datacredito_cols or [
            'tendencia_ingresos',
            'promedio_ingresos_datacredito'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Crear flag si TODAS las variables de Datacredito son nulas
        datacredito_missing = X[self.datacredito_cols].isnull().all(axis=1).astype(int)
        X['sin_historial_datacredito'] = datacredito_missing
        
        return X


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def identify_column_types(df):
    """
    Identifica y clasifica las columnas del DataFrame por tipo.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    
    Retorna:
    --------
    dict : Diccionario con listas de nombres de columnas clasificadas
    """
    # Variables numéricas (excluyendo target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Pago_atiempo' in numeric_cols:
        numeric_cols.remove('Pago_atiempo')
    
    # Variables categóricas
    categorical_nominal = ['tipo_laboral']  # Nominal (sin orden)
    categorical_ordinal = ['tendencia_ingresos']  # Ordinal (con orden)
    
    # Variables temporales
    datetime_cols = ['fecha_prestamo']
    
    # Variables de Datacredito (para flag)
    datacredito_cols = ['tendencia_ingresos', 'promedio_ingresos_datacredito']
    
    return {
        'numeric': numeric_cols,
        'categorical_nominal': categorical_nominal,
        'categorical_ordinal': categorical_ordinal,
        'datetime': datetime_cols,
        'datacredito': datacredito_cols
    }


def create_preprocessing_pipeline(column_types):
    """
    Crea el pipeline completo de preprocesamiento.
    
    Parámetros:
    -----------
    column_types : dict
        Diccionario con clasificación de columnas
    
    Retorna:
    --------
    Pipeline : Pipeline de sklearn con todos los pasos de transformación
    """
    
    # Pipeline para variables numéricas
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para variables categóricas nominales (OneHotEncoder)
    categorical_nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Pipeline para variables categóricas ordinales (OrdinalEncoder)
    # tendencia_ingresos: Creciente > Estable > Decreciente
    categorical_ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            categories=[['Decreciente', 'Estable', 'Creciente']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    
    # ColumnTransformer que aplica las transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, column_types['numeric']),
            ('cat_nominal', categorical_nominal_pipeline, column_types['categorical_nominal']),
            ('cat_ordinal', categorical_ordinal_pipeline, column_types['categorical_ordinal'])
        ],
        remainder='drop'  # Eliminar columnas no especificadas
    )
    
    # Pipeline completo con transformadores personalizados
    full_pipeline = Pipeline([
        ('temporal_features', TemporalFeaturesTransformer()),
        ('datacredito_flag', DatacreditoFlagTransformer(column_types['datacredito'])),
        ('interaction_features', InteractionFeaturesTransformer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline


def fit_transform_pipeline(df, target_col='Pago_atiempo', save_path=None):
    """
    Ajusta y aplica el pipeline de feature engineering completo.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos originales
    target_col : str
        Nombre de la columna objetivo (default: 'Pago_atiempo')
    save_path : str, optional
        Ruta donde guardar el pipeline ajustado
    
    Retorna:
    --------
    tuple : (X_transformed, y, pipeline_fitted, feature_names)
        - X_transformed: Array con features transformadas
        - y: Serie con la variable objetivo
        - pipeline_fitted: Pipeline ajustado
        - feature_names: Lista con nombres de las features finales
    """
    
    print("="*80)
    print("INICIANDO FEATURE ENGINEERING")
    print("="*80)
    
    # Separar features y target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"\n✓ Datos de entrada: {X.shape[0]} registros x {X.shape[1]} features")
    print(f"✓ Variable objetivo: {target_col}")
    
    # Identificar tipos de columnas
    print("\n1. Identificando tipos de columnas...")
    column_types = identify_column_types(df)
    
    print(f"   - Variables numéricas: {len(column_types['numeric'])}")
    print(f"   - Variables categóricas nominales: {len(column_types['categorical_nominal'])}")
    print(f"   - Variables categóricas ordinales: {len(column_types['categorical_ordinal'])}")
    print(f"   - Variables temporales: {len(column_types['datetime'])}")
    
    # Crear pipeline
    print("\n2. Creando pipeline de transformación...")
    pipeline = create_preprocessing_pipeline(column_types)
    
    # Ajustar y transformar
    print("\n3. Ajustando y transformando datos...")
    X_transformed = pipeline.fit_transform(X)
    
    print(f"   ✓ Features transformadas: {X_transformed.shape[1]} features")
    
    # Obtener nombres de features finales
    print("\n4. Obteniendo nombres de features...")
    feature_names = get_feature_names(pipeline, column_types)
    
    print(f"   ✓ Total de features finales: {len(feature_names)}")
    
    # Guardar pipeline si se especificó ruta
    if save_path:
        print(f"\n5. Guardando pipeline en: {save_path}")
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipeline, save_path)
        print("   ✓ Pipeline guardado exitosamente")
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETADO")
    print("="*80)
    
    return X_transformed, y, pipeline, feature_names


def transform_with_pipeline(df, pipeline, target_col='Pago_atiempo'):
    """
    Aplica un pipeline ya ajustado a nuevos datos.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos nuevos
    pipeline : Pipeline
        Pipeline previamente ajustado
    target_col : str
        Nombre de la columna objetivo (default: 'Pago_atiempo')
    
    Retorna:
    --------
    tuple : (X_transformed, y)
        - X_transformed: Array con features transformadas
        - y: Serie con la variable objetivo (si existe)
    """
    
    # Separar features y target
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df.copy()
        y = None
    
    # Transformar
    X_transformed = pipeline.transform(X)
    
    print(f"✓ Datos transformados: {X_transformed.shape[0]} registros x {X_transformed.shape[1]} features")
    
    return X_transformed, y


def get_feature_names(pipeline, column_types):
    """
    Obtiene los nombres de las features después de la transformación.
    
    Parámetros:
    -----------
    pipeline : Pipeline
        Pipeline ajustado
    column_types : dict
        Diccionario con clasificación de columnas (no usado, pero mantenido para compatibilidad)
    
    Retorna:
    --------
    list : Lista con nombres de todas las features finales
    """
    
    # Obtener los nombres directamente del último paso del pipeline (preprocessor)
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        return feature_names.tolist()
    except:
        # Fallback: retornar índices si falla
        n_features = pipeline.named_steps['preprocessor'].transform(pipeline[:-1].transform(X_sample)).shape[1]
        return [f'feature_{i}' for i in range(n_features)]


def get_feature_importance_summary(feature_names, importance_scores):
    """
    Crea un resumen de importancia de features.
    
    Parámetros:
    -----------
    feature_names : list
        Lista de nombres de features
    importance_scores : array
        Scores de importancia (de un modelo)
    
    Retorna:
    --------
    pd.DataFrame : DataFrame con features ordenadas por importancia
    """
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum()
    
    return feature_importance


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal para ejecutar el pipeline completo de feature engineering.
    """
    
    # Importar módulo de carga de datos
    from cargar_datos import load_data
    
    # Cargar datos
    print("Cargando datos...")
    df = load_data('../../Base_de_datos.xlsx')
    
    # Aplicar feature engineering
    X_transformed, y, pipeline, feature_names = fit_transform_pipeline(
        df,
        target_col='Pago_atiempo',
        save_path=None  # No guardar automáticamente
    )
    
    # Crear DataFrame con features transformadas
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Mostrar información
    print("\n" + "="*80)
    print("RESUMEN DE FEATURES TRANSFORMADAS")
    print("="*80)
    print(f"\nShape final: {df_transformed.shape}")
    print(f"\nPrimeras 5 features:")
    print(feature_names[:5])
    print(f"\nÚltimas 5 features:")
    print(feature_names[-5:])
    
    print("\nPrimeros registros transformados:")
    print(df_transformed.head())
    
    print("\n" + "="*80)
    print("✓ FEATURE ENGINEERING COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    return df_transformed, y, pipeline, feature_names


if __name__ == "__main__":
    main()
