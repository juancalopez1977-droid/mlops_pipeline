"""
model_deploy.py
===============
API REST con FastAPI para despliegue del modelo de riesgo crediticio.

Este módulo implementa:
- Modelos Pydantic para validación de entrada/salida
- Endpoints para predicción individual y por lotes
- Carga automática del modelo y pipeline entrenados
- Manejo de errores y respuestas HTTP

Autor: Juan Carlos López S.
Fecha: Mayo 2026
Versión: 1.3.0
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import glob
import os
import sys

# Agregar directorio src al path para que joblib encuentre ft_engineering
# (el pipeline contiene transformadores personalizados definidos en ft_engineering)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Importar ft_engineering para que esté disponible cuando joblib deserialice el pipeline
import ft_engineering

# ============================================================================
# MODELOS PYDANTIC PARA VALIDACIÓN
# ============================================================================

class PredictionInput(BaseModel):
    """
    Modelo de entrada para predicción individual.
    Contiene todos los campos necesarios para evaluar riesgo crediticio.
    """
    # Variables numéricas (18 campos)
    tipo_credito: float = Field(..., description="Tipo de crédito (numérico)")
    capital_prestado: float = Field(..., ge=0, description="Monto del préstamo")
    plazo_meses: int = Field(..., ge=1, description="Plazo en meses")
    edad_cliente: int = Field(..., ge=18, le=100, description="Edad del cliente")
    salario_cliente: float = Field(..., ge=0, description="Salario mensual")
    total_otros_prestamos: float = Field(..., ge=0, description="Total de otros préstamos")
    cuota_pactada: float = Field(..., ge=0, description="Cuota mensual pactada")
    puntaje: float = Field(..., description="Puntaje interno")
    puntaje_datacredito: Optional[float] = Field(None, description="Puntaje Datacrédito")
    cant_creditosvigentes: int = Field(..., ge=0, description="Cantidad de créditos vigentes")
    huella_consulta: int = Field(..., ge=0, description="Huellas de consulta")
    saldo_mora: float = Field(..., ge=0, description="Saldo en mora")
    saldo_total: float = Field(..., ge=0, description="Saldo total")
    saldo_principal: float = Field(..., ge=0, description="Saldo del principal")
    saldo_mora_codeudor: float = Field(..., ge=0, description="Saldo mora codeudor")
    creditos_sectorFinanciero: int = Field(..., ge=0, description="Créditos sector financiero")
    creditos_sectorCooperativo: int = Field(..., ge=0, description="Créditos sector cooperativo")
    creditos_sectorReal: int = Field(..., ge=0, description="Créditos sector real")
    promedio_ingresos_datacredito: Optional[float] = Field(None, description="Promedio ingresos Datacrédito")
    
    # Variables categóricas (3 campos)
    tipo_laboral: str = Field(..., description="Tipo laboral (Dependiente/Independiente)")
    tendencia_ingresos: str = Field(..., description="Tendencia de ingresos (Creciente/Estable/Decreciente)")
    fecha_prestamo: str = Field(..., description="Fecha del préstamo (YYYY-MM-DD)")
    
    # Variable categórica opcional
    Datacredito: Optional[str] = Field(None, description="Estado Datacrédito")

    class Config:
        schema_extra = {
            "example": {
                "tipo_credito": 1.5,
                "capital_prestado": 5000000,
                "plazo_meses": 24,
                "edad_cliente": 35,
                "salario_cliente": 2500000,
                "total_otros_prestamos": 1000000,
                "cuota_pactada": 250000,
                "puntaje": 750,
                "puntaje_datacredito": 700,
                "cant_creditosvigentes": 2,
                "huella_consulta": 3,
                "saldo_mora": 0,
                "saldo_total": 3000000,
                "saldo_principal": 2800000,
                "saldo_mora_codeudor": 0,
                "creditos_sectorFinanciero": 1,
                "creditos_sectorCooperativo": 1,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 2400000,
                "tipo_laboral": "Dependiente",
                "tendencia_ingresos": "Creciente",
                "fecha_prestamo": "2026-05-27",
                "Datacredito": "Al dia"
            }
        }


class PredictionOutput(BaseModel):
    """
    Modelo de salida con resultado de predicción.
    """
    probabilidad_pago: float = Field(..., ge=0, le=1, description="Probabilidad de pago a tiempo")
    clase_predicha: int = Field(..., ge=0, le=1, description="Clase predicha (0=No paga, 1=Paga)")
    riesgo: str = Field(..., description="Nivel de riesgo (BAJO/MEDIO/ALTO)")
    confianza: float = Field(..., ge=0, le=1, description="Confianza de la predicción")

    class Config:
        schema_extra = {
            "example": {
                "probabilidad_pago": 0.95,
                "clase_predicha": 1,
                "riesgo": "BAJO",
                "confianza": 0.95
            }
        }


class BatchPredictionInput(BaseModel):
    """
    Modelo para predicciones por lotes.
    """
    predictions: List[PredictionInput] = Field(..., description="Lista de predicciones a procesar")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "tipo_credito": 1.5,
                        "capital_prestado": 5000000,
                        "plazo_meses": 24,
                        "edad_cliente": 35,
                        "salario_cliente": 2500000,
                        "total_otros_prestamos": 1000000,
                        "cuota_pactada": 250000,
                        "puntaje": 750,
                        "puntaje_datacredito": 700,
                        "cant_creditosvigentes": 2,
                        "huella_consulta": 3,
                        "saldo_mora": 0,
                        "saldo_total": 3000000,
                        "saldo_principal": 2800000,
                        "saldo_mora_codeudor": 0,
                        "creditos_sectorFinanciero": 1,
                        "creditos_sectorCooperativo": 1,
                        "creditos_sectorReal": 0,
                        "promedio_ingresos_datacredito": 2400000,
                        "tipo_laboral": "Dependiente",
                        "tendencia_ingresos": "Creciente",
                        "fecha_prestamo": "2026-05-27",
                        "Datacredito": "Al dia"
                    }
                ]
            }
        }


class BatchPredictionOutput(BaseModel):
    """
    Modelo de salida para predicciones por lotes.
    """
    predictions: List[PredictionOutput] = Field(..., description="Lista de resultados")
    total_processed: int = Field(..., description="Total de registros procesados")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "probabilidad_pago": 0.95,
                        "clase_predicha": 1,
                        "riesgo": "BAJO",
                        "confianza": 0.95
                    }
                ],
                "total_processed": 1
            }
        }


# ============================================================================
# CARGA DE MODELO Y PIPELINE
# ============================================================================

def load_model_artifacts():
    """
    Carga el modelo entrenado, pipeline de preprocesamiento y metadata.
    
    Returns:
        tuple: (model, pipeline, metadata)
    """
    try:
        # Obtener directorio del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        # Buscar archivo de modelo (puede variar por timestamp)
        model_pattern = os.path.join(models_dir, 'best_model_*.pkl')
        model_files = glob.glob(model_pattern)
        if not model_files:
            raise FileNotFoundError(f"No se encontró archivo de modelo en {models_dir}/")
        
        model_path = model_files[0]  # Tomar el primer modelo encontrado
        print(f"Cargando modelo desde: {model_path}")
        model = joblib.load(model_path)
        
        # Cargar pipeline
        pipeline_path = os.path.join(models_dir, 'preprocessing_pipeline.pkl')
        print(f"Cargando pipeline desde: {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        
        # Cargar metadata
        metadata_path = os.path.join(models_dir, 'model_metadata.json')
        print(f"Cargando metadata desde: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print("✓ Modelo, pipeline y metadata cargados exitosamente")
        return model, pipeline, metadata
    
    except Exception as e:
        print(f"ERROR al cargar modelo: {str(e)}")
        raise


# Cargar modelo al iniciar la aplicación
print("\n" + "="*80)
print("INICIANDO API DE PREDICCIÓN DE RIESGO CREDITICIO")
print("="*80)

model, pipeline, metadata = load_model_artifacts()

print(f"\nModelo cargado: {metadata.get('model_name', 'Unknown')}")
print(f"F1-Score: {metadata.get('metrics', {}).get('f1_score', 'N/A')}")
print(f"Features: {len(metadata.get('features_used', []))}")
print(f"Entrenado: {metadata.get('timestamp', 'N/A')}")
print("="*80 + "\n")


# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title="MLOps Credit Risk Prediction API",
    description="API REST para predicción de riesgo crediticio usando modelo de Machine Learning",
    version="1.3.0",
    contact={
        "name": "Juan Carlos López S.",
        "email": "juanc@example.com"
    }
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
def root():
    """
    Endpoint raíz con información básica de la API.
    """
    return {
        "service": "Credit Risk Prediction API",
        "version": "1.3.0",
        "model": metadata.get("model_name", "Unknown"),
        "f1_score": metadata.get("metrics", {}).get("f1_score"),
        "accuracy": metadata.get("metrics", {}).get("accuracy"),
        "trained_date": metadata.get("timestamp"),
        "features_count": len(metadata.get("features_used", [])),
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["General"])
def health_check():
    """
    Endpoint de verificación de salud del servicio.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
def predict(input_data: PredictionInput):
    """
    Realiza una predicción individual de riesgo crediticio.
    
    Args:
        input_data: Datos del cliente en formato PredictionInput
    
    Returns:
        PredictionOutput: Resultado con probabilidad, clase, riesgo y confianza
    
    Raises:
        HTTPException: Si ocurre un error durante la predicción
    """
    try:
        # Convertir a DataFrame (1 registro)
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])
        
        # Aplicar pipeline de preprocesamiento
        X_transformed = pipeline.transform(df)
        
        # Realizar predicción
        pred_class = model.predict(X_transformed)[0]
        pred_proba = model.predict_proba(X_transformed)[0]
        
        # Extraer probabilidad de pago (clase 1)
        prob_pago = float(pred_proba[1])
        
        # Clasificar nivel de riesgo basado en probabilidad
        if prob_pago >= 0.7:
            riesgo = "BAJO"
        elif prob_pago >= 0.4:
            riesgo = "MEDIO"
        else:
            riesgo = "ALTO"
        
        # Calcular confianza (máxima probabilidad)
        confianza = float(max(pred_proba))
        
        return PredictionOutput(
            probabilidad_pago=prob_pago,
            clase_predicha=int(pred_class),
            riesgo=riesgo,
            confianza=confianza
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
def predict_batch(input_data: BatchPredictionInput):
    """
    Realiza predicciones por lotes (múltiples registros).
    
    Args:
        input_data: Lista de registros en formato BatchPredictionInput
    
    Returns:
        BatchPredictionOutput: Lista de resultados y total procesado
    
    Raises:
        HTTPException: Si ocurre un error durante las predicciones
    """
    try:
        # Validar que haya al menos un registro
        if not input_data.predictions:
            raise HTTPException(
                status_code=400,
                detail="La lista de predicciones no puede estar vacía"
            )
        
        # Convertir lista de inputs a DataFrame
        data_list = [item.dict() for item in input_data.predictions]
        df = pd.DataFrame(data_list)
        
        # Aplicar pipeline de preprocesamiento
        X_transformed = pipeline.transform(df)
        
        # Realizar predicciones en batch
        pred_classes = model.predict(X_transformed)
        pred_probas = model.predict_proba(X_transformed)
        
        # Construir lista de resultados
        results = []
        for i in range(len(df)):
            prob_pago = float(pred_probas[i][1])
            
            # Clasificar riesgo
            if prob_pago >= 0.7:
                riesgo = "BAJO"
            elif prob_pago >= 0.4:
                riesgo = "MEDIO"
            else:
                riesgo = "ALTO"
            
            confianza = float(max(pred_probas[i]))
            
            results.append(PredictionOutput(
                probabilidad_pago=prob_pago,
                clase_predicha=int(pred_classes[i]),
                riesgo=riesgo,
                confianza=confianza
            ))
        
        return BatchPredictionOutput(
            predictions=results,
            total_processed=len(results)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante las predicciones batch: {str(e)}"
        )


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("INICIANDO SERVIDOR UVICORN")
    print("="*80)
    print("Accede a la documentación interactiva en: http://localhost:8000/docs")
    print("Accede a la documentación alternativa en: http://localhost:8000/redoc")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
