"""
model_training_evaluation.py
=============================
Módulo de entrenamiento y evaluación de modelos supervisados para predicción de riesgo crediticio.

Este módulo implementa:
- Funciones reutilizables: summarize_classification() y build_model()
- División de datos train/test
- Entrenamiento de múltiples modelos supervisados
- Manejo del desbalance de clases
- Evaluación comparativa con métricas
- Visualizaciones comparativas
- Tabla resumen de performance

Autor: Juan Carlos López S.
Fecha: Mayo 2026
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def summarize_classification(y_true, y_pred, y_pred_proba=None, model_name="Modelo"):
    """
    Genera un resumen completo de métricas de clasificación.
    
    Parámetros:
    -----------
    y_true : array-like
        Valores reales del target
    y_pred : array-like
        Predicciones del modelo
    y_pred_proba : array-like, optional
        Probabilidades predichas para la clase positiva
    model_name : str
        Nombre del modelo para el reporte
    
    Retorna:
    --------
    dict : Diccionario con todas las métricas calculadas
    """
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calcular ROC AUC si se proporcionan probabilidades
    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Especificidad
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Crear diccionario de métricas
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    # Imprimir resumen
    print(f"\n{'='*80}")
    print(f"RESUMEN DE CLASIFICACIÓN: {model_name}")
    print(f"{'='*80}")
    print(f"\nMétricas Principales:")
    print(f"  • Accuracy:    {accuracy:.4f}")
    print(f"  • Precision:   {precision:.4f}")
    print(f"  • Recall:      {recall:.4f}")
    print(f"  • F1-Score:    {f1:.4f}")
    if roc_auc is not None:
        print(f"  • ROC AUC:     {roc_auc:.4f}")
    print(f"  • Specificity: {specificity:.4f}")
    
    print(f"\nMatriz de Confusión:")
    print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
    print(f"  FN: {fn:5d}  |  TP: {tp:5d}")
    
    return metrics


def build_model(model, X_train, y_train, X_test, y_test, model_name="Modelo", use_proba=True):
    """
    Entrena y evalúa un modelo de clasificación de forma estandarizada.
    
    Parámetros:
    -----------
    model : estimator
        Modelo de sklearn a entrenar
    X_train : array-like
        Features de entrenamiento
    y_train : array-like
        Target de entrenamiento
    X_test : array-like
        Features de prueba
    y_test : array-like
        Target de prueba
    model_name : str
        Nombre del modelo
    use_proba : bool
        Si usar predict_proba para ROC AUC
    
    Retorna:
    --------
    tuple : (modelo_entrenado, métricas_dict, predicciones, probabilidades)
    """
    
    print(f"\n{'='*80}")
    print(f"ENTRENANDO: {model_name}")
    print(f"{'='*80}")
    
    # Entrenar modelo
    model.fit(X_train, y_train)
    print("✓ Modelo entrenado exitosamente")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Probabilidades (si están disponibles)
    y_pred_proba = None
    if use_proba and hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluar
    metrics = summarize_classification(y_test, y_pred, y_pred_proba, model_name)
    
    return model, metrics, y_pred, y_pred_proba


def prepare_data(df, target_col='Pago_atiempo', test_size=0.25, random_state=42, apply_smote=False):
    """
    Prepara los datos aplicando feature engineering y división train/test.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos crudos
    target_col : str
        Nombre de la columna objetivo
    test_size : float
        Proporción de datos para test
    random_state : int
        Semilla aleatoria
    apply_smote : bool
        Si aplicar SMOTE para balancear clases
    
    Retorna:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    
    from ft_engineering import fit_transform_pipeline
    
    print("="*80)
    print("PREPARACIÓN DE DATOS")
    print("="*80)
    
    # Aplicar feature engineering
    print("\n1. Aplicando feature engineering...")
    X_transformed, y, pipeline, feature_names = fit_transform_pipeline(df, target_col, save_path=None)
    
    print(f"   ✓ Features transformadas: {X_transformed.shape[1]} features")
    print(f"   ✓ Total de registros: {X_transformed.shape[0]}")
    
    # División train/test
    print(f"\n2. Dividiendo datos (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"   ✓ Train: {X_train.shape[0]} registros")
    print(f"   ✓ Test:  {X_test.shape[0]} registros")
    
    # Verificar balance de clases
    train_balance = y_train.value_counts(normalize=True)
    print(f"\n3. Balance de clases en train:")
    print(f"   • Clase 0: {train_balance[0]:.2%}")
    print(f"   • Clase 1: {train_balance[1]:.2%}")
    
    # Aplicar SMOTE si se solicita
    if apply_smote:
        print(f"\n4. Aplicando SMOTE para balancear clases...")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        train_balance_after = pd.Series(y_train).value_counts(normalize=True)
        print(f"   ✓ Balance después de SMOTE:")
        print(f"     • Clase 0: {train_balance_after[0]:.2%}")
        print(f"     • Clase 1: {train_balance_after[1]:.2%}")
        print(f"   ✓ Nuevo tamaño train: {X_train.shape[0]} registros")
    
    print("\n" + "="*80)
    print("DATOS PREPARADOS")
    print("="*80)
    
    return X_train, X_test, y_train, y_test


def train_multiple_models(X_train, y_train, X_test, y_test, use_smote_weights=False):
    """
    Entrena y evalúa múltiples modelos de clasificación.
    
    Parámetros:
    -----------
    X_train, y_train : array-like
        Datos de entrenamiento
    X_test, y_test : array-like
        Datos de prueba
    use_smote_weights : bool
        Si usar class_weight='balanced' en modelos que lo soporten
    
    Retorna:
    --------
    dict : Diccionario con resultados de todos los modelos
    """
    
    results = {}
    
    # Configurar modelos
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced' if use_smote_weights else None
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            class_weight='balanced' if use_smote_weights else None
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            class_weight='balanced' if use_smote_weights else None
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Entrenar cada modelo
    for model_name, model in models.items():
        trained_model, metrics, y_pred, y_pred_proba = build_model(
            model, X_train, y_train, X_test, y_test, model_name
        )
        
        results[model_name] = {
            'model': trained_model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    return results


def create_comparison_table(results):
    """
    Crea una tabla comparativa de métricas de todos los modelos.
    
    Parámetros:
    -----------
    results : dict
        Diccionario con resultados de múltiples modelos
    
    Retorna:
    --------
    pd.DataFrame : Tabla comparativa
    """
    
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Modelo': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A',
            'Specificity': f"{metrics['specificity']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE MODELOS")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80)
    
    return df_comparison


def plot_model_comparison(results, y_test):
    """
    Genera gráficos comparativos para todos los modelos.
    
    Parámetros:
    -----------
    results : dict
        Diccionario con resultados de múltiples modelos
    y_test : array-like
        Target real de prueba
    """
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # 1. Comparación de métricas principales
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparación de Modelos - Métricas Principales', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        models = list(results.keys())
        values = [results[model]['metrics'][metric] for model in models]
        
        bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} por Modelo')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Curvas ROC
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, result in results.items():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            roc_auc = result['metrics']['roc_auc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. Matrices de confusión
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Matrices de Confusión por Modelo', fontsize=16, fontweight='bold')
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx // 3, idx % 3]
        
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(model_name, fontweight='bold')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicción')
    
    # Ocultar ejes sobrantes si hay menos de 6 modelos
    for idx in range(n_models, 6):
        axes[idx // 3, idx % 3].axis('off')
    
    plt.tight_layout()
    plt.show()


def select_best_model(results, metric='f1_score'):
    """
    Selecciona el mejor modelo según una métrica específica.
    
    Parámetros:
    -----------
    results : dict
        Diccionario con resultados de múltiples modelos
    metric : str
        Métrica a usar para seleccionar el mejor modelo
    
    Retorna:
    --------
    tuple : (nombre_mejor_modelo, mejor_modelo, métricas)
    """
    
    best_score = -1
    best_model_name = None
    
    for model_name, result in results.items():
        score = result['metrics'][metric]
        if score is not None and score > best_score:
            best_score = score
            best_model_name = model_name
    
    best_result = results[best_model_name]
    
    print("\n" + "="*80)
    print(f"MEJOR MODELO SEGÚN {metric.upper()}")
    print("="*80)
    print(f"\nModelo seleccionado: {best_model_name}")
    print(f"{metric}: {best_score:.4f}")
    print("\nMétricas completas:")
    for key, value in best_result['metrics'].items():
        if key != 'model_name' and value is not None:
            print(f"  • {key}: {value:.4f}" if isinstance(value, float) else f"  • {key}: {value}")
    print("="*80)
    
    return best_model_name, best_result['model'], best_result['metrics']


def main():
    """
    Función principal para ejecutar el pipeline completo de modelamiento.
    """
    
    # Importar módulo de carga de datos
    from cargar_datos import load_data
    
    # Cargar datos
    print("Cargando datos...")
    df = load_data('../../Base_de_datos.xlsx')
    
    # Preparar datos CON SMOTE
    print("\n" + "="*80)
    print("ESTRATEGIA: APLICAR SMOTE")
    print("="*80)
    
    X_train, X_test, y_train, y_test = prepare_data(
        df, 
        target_col='Pago_atiempo',
        test_size=0.25,
        random_state=42,
        apply_smote=True
    )
    
    # Entrenar modelos
    results = train_multiple_models(X_train, y_train, X_test, y_test, use_smote_weights=False)
    
    # Crear tabla comparativa
    comparison_table = create_comparison_table(results)
    
    # Generar gráficos comparativos
    plot_model_comparison(results, y_test)
    
    # Seleccionar mejor modelo
    best_name, best_model, best_metrics = select_best_model(results, metric='f1_score')
    
    print("\n" + "="*80)
    print("✓ PROCESO DE MODELAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    return results, comparison_table, best_model


if __name__ == "__main__":
    results, comparison_table, best_model = main()
