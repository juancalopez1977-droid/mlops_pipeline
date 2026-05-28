# Dockerfile para MLOps Credit Risk Prediction API
# Versión: 1.3.0
# Autor: Juan Carlos López S.
# Fecha: Mayo 2026

# Imagen base de Python 3.11
FROM python:3.11-slim

# Metadata de la imagen
LABEL maintainer="juanc@example.com"
LABEL version="1.3.0"
LABEL description="API REST para predicción de riesgo crediticio con FastAPI y ML"

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivo de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente del proyecto
COPY mlops_pipeline/ ./mlops_pipeline/
COPY Base_de_datos.xlsx .

# Copiar modelos entrenados (excluidos por .dockerignore, se copian explícitamente)
COPY mlops_pipeline/src/models/*.pkl ./mlops_pipeline/src/models/
COPY mlops_pipeline/src/models/*.json ./mlops_pipeline/src/models/

# Exponer puerto 8000 para la API
EXPOSE 8000

# Variable de entorno para modo producción
ENV PYTHONUNBUFFERED=1

# Healthcheck para verificar que la API está respondiendo
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando de inicio: ejecutar servidor uvicorn
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
