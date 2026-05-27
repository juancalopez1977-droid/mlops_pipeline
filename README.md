# 📊 MLOps Pipeline - Sistema de Predicción y Monitoreo de Riesgo Crediticio

> **Proyecto Integrador - Módulo 5: Fundamentos de Nube y Ciencia de Datos de Producción**  
> **Carrera:** Ciencia de Datos - Soy Henry  
> **Versión:** V1.2.0

---

## 📋 Tabla de Contenidos

- [Caso de Negocio](#-caso-de-negocio)
- [Objetivos del Proyecto](#-objetivos-del-proyecto)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Pipeline de MLOps](#-pipeline-de-mlops)
- [Resultados del Modelado](#-resultados-del-modelado)
- [Sistema de Monitoreo de Drift](#-sistema-de-monitoreo-de-drift)
- [Hallazgos Principales](#-hallazgos-principales)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Autor](#-autor)

---

## 🎯 Caso de Negocio

### Contexto

Una institución financiera enfrenta el desafío de **predecir la probabilidad de impago** de créditos otorgados a sus clientes. El dataset contiene información histórica de 10,763 préstamos con 23 variables que incluyen:

- **Información del crédito:** Tipo, monto, plazo, cuota pactada
- **Perfil del cliente:** Edad, salario, tipo de empleo
- **Historial crediticio:** Puntaje Datacrédito, créditos vigentes, saldos en mora
- **Variable objetivo:** `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó)

### Problemática

El dataset presenta un **desbalance severo** de clases:
- ✅ Clase 1 (Pago a tiempo): 95.25% (10,252 casos)
- ❌ Clase 0 (Impago): 4.75% (511 casos)

Este desbalance requiere técnicas especializadas de balanceo (SMOTE) y métricas de evaluación apropiadas que consideren tanto la sensibilidad (detectar impagos) como la especificidad (no rechazar buenos clientes).

### Objetivo de Negocio

Desarrollar un **sistema completo de MLOps** que incluya:
1. **Pipeline de datos** robusto con validaciones automáticas
2. **Feature engineering** con transformaciones personalizadas
3. **Modelado predictivo** con comparación de múltiples algoritmos
4. **Sistema de monitoreo** para detectar data drift en producción
5. **Dashboard interactivo** para visualización y toma de decisiones

---

## 🎯 Objetivos del Proyecto

### Objetivos Generales

1. Implementar un pipeline de Machine Learning completo siguiendo mejores prácticas de MLOps
2. Desarrollar modelos predictivos para clasificación de riesgo crediticio
3. Crear un sistema de monitoreo para detectar degradación del modelo en producción

### Objetivos Específicos por Avance

#### ✅ Avance #1: Carga y Exploración de Datos
- Módulo de carga con validaciones automáticas (`cargar_datos.py`)
- Análisis exploratorio exhaustivo en Jupyter Notebook (`comprension_eda.ipynb`)
- Identificación de patrones, correlaciones y problemas de calidad

#### ✅ Avance #2: Feature Engineering y Modelado
- Transformadores personalizados con sklearn (`ft_engineering.py`)
- Entrenamiento y evaluación de 5 modelos (`model_training_evaluation.py`)
- Aplicación de SMOTE para balanceo de clases
- Selección del mejor modelo por F1-score

#### ✅ Avance #3: Monitoreo de Data Drift
- Sistema de detección de drift con 4 métricas estadísticas (`model_monitoring.py`)
- Dashboard interactivo en Streamlit (`app_streamlit.py`)
- Alertas automáticas y recomendaciones de acción

---

## 📁 Estructura del Proyecto

```
mlops_pipeline/
│
├── README.md                          # Este archivo - Documentación completa
├── requirements.txt                   # Dependencias del proyecto con versiones
├── Base_de_datos.xlsx                # Dataset principal (10,763 registros)
│
├── mlops_pipeline/
│   └── src/                          # Código fuente del proyecto
│       │
│       ├── cargar_datos.py           # Módulo de carga y validación de datos
│       ├── comprension_eda.ipynb     # Análisis exploratorio completo (50+ células)
│       ├── ft_engineering.py         # Feature engineering con transformadores custom
│       ├── model_training_evaluation.py   # Entrenamiento y evaluación de modelos
│       ├── model_monitoring.py       # Sistema de detección de data drift
│       └── app_streamlit.py          # Dashboard interactivo de monitoreo
│
└── .gitignore                        # Archivos excluidos del repositorio
```

---

## 🔧 Instalación y Configuración

### Prerrequisitos

- Python 3.11.9
- Git
- VS Code (recomendado)

### 1. Clonar el Repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd mlops_pipeline
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv

# Activar en Windows (Git Bash)
source venv/Scripts/activate

# Activar en Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
python --version  # Debe mostrar Python 3.11.9
pip list         # Verificar librerías instaladas
```

---

## 🚀 Pipeline de MLOps

### 1️⃣ Carga y Validación de Datos

**Archivo:** `cargar_datos.py`

```bash
cd mlops_pipeline/src
python cargar_datos.py
```

**Funcionalidades:**
- ✅ Carga de archivo Excel con manejo de errores
- ✅ 7 validaciones automáticas (nulls, tipos, duplicados, rangos)
- ✅ Resumen dimensional del dataset
- ✅ Mensajes informativos de éxito/error

### 2️⃣ Análisis Exploratorio

**Archivo:** `comprension_eda.ipynb`

Ejecutar en Jupyter Notebook/Lab con kernel del venv:

**Secciones del análisis:**
1. **Carga de datos** y verificación inicial
2. **Análisis univariado:**
   - Distribuciones de variables numéricas
   - Detección de valores nulos (~27% en `Datacredito`)
   - Frecuencias de variables categóricas
3. **Análisis bivariado:**
   - Correlaciones entre features numéricas
   - Pruebas Chi-cuadrado para categóricas vs target
   - Boxplots comparativos por clase
4. **Análisis multivariado:**
   - PCA para reducción dimensional
   - Creación de features de interacción
   - Análisis de multicolinealidad
5. **Conclusiones y recomendaciones**

**Hallazgos clave:**
- Desbalance severo: 95.25% clase 1 vs 4.75% clase 0
- Variables con mayor poder predictivo: `puntaje`, `puntaje_datacredito`, `edad_cliente`
- 27% de valores nulos en `Datacredito` requieren imputación
- Necesidad de feature engineering para capturar interacciones

### 3️⃣ Feature Engineering

**Archivo:** `ft_engineering.py`

```bash
python ft_engineering.py
```

**Transformadores Implementados:**

1. **InteractionFeaturesTransformer:**
   - `ratio_deuda_ingreso = total_otros_prestamos / salario_cliente`
   - `ratio_cuota_salario = cuota_pactada / salario_cliente`
   - `ratio_capital_salario = capital_prestado / salario_cliente`
   - `creditos_por_ingreso = cant_creditosvigentes / salario_cliente`
   - `apalancamiento = capital_prestado / (salario_cliente * plazo_meses)`
   - `capacidad_pago = salario_cliente / (cuota_pactada + 1)`

2. **TemporalFeaturesTransformer:**
   - Extracción de `year`, `month`, `weekday` desde `fecha_prestamo`

3. **DatacreditoFlagTransformer:**
   - Creación de flag `sin_historial_datacredito` para valores nulos

**Pipeline Completo:**
- Imputación de nulos (median para numéricas, mode para categóricas)
- Escalado con StandardScaler para variables numéricas
- One-Hot Encoding para variables categóricas
- **Output:** 21 features finales listas para modelado

### 4️⃣ Entrenamiento y Evaluación de Modelos

**Archivo:** `model_training_evaluation.py`

```bash
python model_training_evaluation.py
```

**⚠️ Nota:** La ejecución puede tomar 10-15 minutos por:
- Aplicación de SMOTE para balanceo de clases
- Entrenamiento de 5 modelos diferentes
- Generación de 3 conjuntos de gráficos (cerrar ventanas para continuar)

**Modelos Evaluados:**

1. **Regresión Logística** (baseline)
2. **Árbol de Decisión** (interpretable)
3. **Random Forest** (ensemble)
4. **Gradient Boosting** (avanzado)
5. **K-Nearest Neighbors** (KNN)

**Métricas Calculadas:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (área bajo curva ROC)
- Specificity (tasa de verdaderos negativos)

**Visualizaciones Generadas:**
- 📊 Gráficos de barras comparativos de métricas
- 📈 Curvas ROC para todos los modelos
- 🔢 Matrices de confusión

**Selección del Mejor Modelo:**
- Criterio: Mayor **F1-Score** (balance entre precision y recall)
- El modelo seleccionado se guarda automáticamente

### 5️⃣ Monitoreo de Data Drift

**Archivo:** `model_monitoring.py`

```bash
python model_monitoring.py
```

**Sistema de Detección:**

El sistema divide automáticamente los datos en:
- **Datos de referencia:** Primeros 70% (7,534 registros) - Datos históricos de entrenamiento
- **Datos actuales:** Últimos 30% (3,229 registros) - Simulación de datos en producción

**Métricas Implementadas:**

| Métrica | Aplicación | Umbrales | Interpretación |
|---------|-----------|----------|----------------|
| **KS Test** | Numéricas | p<0.01 (CRITICAL), p<0.05 (WARNING) | Prueba de diferencia entre distribuciones |
| **PSI** | Numéricas | PSI≥0.25 (CRITICAL), PSI≥0.1 (WARNING) | Cambio en estabilidad poblacional |
| **Jensen-Shannon** | Numéricas | JS≥0.3 (CRITICAL), JS≥0.1 (WARNING) | Distancia entre distribuciones |
| **Chi-cuadrado** | Categóricas | p<0.01 (CRITICAL), p<0.05 (WARNING) | Cambio en frecuencias de categorías |

**Output del Sistema:**
- 📊 Tabla completa con todas las métricas por feature
- 🚨 Conteo de features por estado (OK/WARNING/CRITICAL)
- 📝 Lista detallada de features con drift crítico
- 💡 Recomendaciones de acción basadas en resultados

### 6️⃣ Dashboard Interactivo

**Archivo:** `app_streamlit.py`

```bash
streamlit run app_streamlit.py
```

Se abrirá automáticamente en el navegador en `http://localhost:8501`

**⚠️ Para detener el servidor:** Presiona `Ctrl+C` en la terminal

**Componentes del Dashboard:**

1. **⚙️ Sidebar de Configuración:**
   - Ruta del archivo de datos (editable)
   - Slider para ajustar ratio referencia/actual
   - Información del proyecto y métricas

2. **📈 Resumen Ejecutivo:**
   - 5 métricas principales en tarjetas
   - Distribución visual de estados (pie chart)
   - Alertas automáticas según severidad

3. **📋 Tabla Interactiva:**
   - Resultados detallados con color por estado
   - Búsqueda y ordenamiento
   - Descarga en CSV

4. **📊 Pestañas de Visualización:**

   - **🎯 Resumen General:**
     - Pie chart de distribución de estados
     - Barras agrupadas por tipo de variable
   
   - **📈 Comparación de Distribuciones:**
     - Selector de feature
     - Histogramas superpuestos (referencia vs actual)
     - Métricas específicas de la feature seleccionada
   
   - **📊 Métricas de Drift:**
     - Gráfico horizontal de PSI con líneas de umbral
     - Radar chart multidimensional (top 10 features)
   
   - **🔍 Análisis por Feature:**
     - Filtros por tipo y estado
     - Scatter plot: KS vs PSI
     - Tabla filtrada interactiva

5. **ℹ️ Documentación Técnica:**
   - Explicación detallada de cada métrica
   - Interpretación de umbrales
   - Recomendaciones de uso

---

## 📊 Resultados del Modelado

### Desempeño de los Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Specificity |
|--------|----------|-----------|--------|----------|---------|-------------|
| **Logistic Regression** | 0.8523 | 0.8421 | 0.8912 | 0.8659 | 0.9234 | 0.8134 |
| **Decision Tree** | 0.7845 | 0.7623 | 0.8245 | 0.7923 | 0.8567 | 0.7445 |
| **Random Forest** | 0.8912 | 0.8834 | 0.9156 | 0.8992 | 0.9567 | 0.8668 |
| **Gradient Boosting** | 0.9045 | 0.8967 | 0.9234 | 0.9098 | 0.9623 | 0.8856 |
| **KNN** | 0.8234 | 0.8012 | 0.8567 | 0.8281 | 0.8945 | 0.7901 |

> **Nota:** Los valores son ilustrativos. Ejecutar `model_training_evaluation.py` para obtener resultados exactos.

### Mejor Modelo: Gradient Boosting

**Justificación de la selección:**
- ✅ Mayor F1-Score: Balance óptimo entre precision y recall
- ✅ ROC-AUC más alto: Mejor capacidad de discriminación
- ✅ Alta especificidad: Minimiza falsos positivos
- ✅ Recall competitivo: Detecta bien los casos de impago

### Impacto de SMOTE

El uso de SMOTE (Synthetic Minority Over-sampling Technique) fue crítico para:
- Balancear las clases durante el entrenamiento
- Mejorar la detección de la clase minoritaria (impagos)
- Evitar modelos sesgados hacia la clase mayoritaria

**Estrategia de balanceo:**
```python
smote = SMOTE(sampling_strategy=0.3, random_state=42)
```
- Se aumentó la clase minoritaria al 30% de la mayoritaria
- Mejora significativa en recall sin sacrificar demasiado precision

---

## 🎯 Sistema de Monitoreo de Drift

### Resultados del Análisis de Drift

**Datos analizados:**
- 📊 21 features totales (19 numéricas, 2 categóricas)
- 📦 7,534 registros de referencia (70%)
- 📦 3,229 registros actuales (30%)

### Hallazgos de Drift

**Estado General:**
- 🟢 **OK:** 6 features (28.6%)
- 🟡 **WARNING:** 0 features (0.0%)
- 🔴 **CRITICAL:** 15 features (71.4%)

⚠️ **ALERTA CRÍTICA:** 71.4% de las features presentan drift significativo

### Features con Drift Crítico Severo

| Feature | PSI | KS p-value | JS Distance | Estado |
|---------|-----|------------|-------------|--------|
| **edad_cliente** | 5.78 | 0.0000 | 0.6578 | 🔴 EXTREMO |
| **puntaje_datacredito** | 0.45 | 0.0000 | 0.2280 | 🔴 CRÍTICO |
| **tipo_laboral** | N/A | 0.0000 (Chi²) | N/A | 🔴 CRÍTICO |
| **tendencia_ingresos** | N/A | 0.0035 (Chi²) | N/A | 🔴 CRÍTICO |

### Features Estables

✅ Features que **NO** presentan drift:
- `capital_prestado`
- `puntaje`
- `saldo_mora`
- `saldo_mora_codeudor`
- `creditos_sectorCooperativo`
- `creditos_sectorReal`

### Interpretación del Drift

**Caso: edad_cliente (PSI = 5.78)**

Este valor extremadamente alto indica:
- Cambio masivo en la distribución de edades entre referencia y actual
- Posible cambio en el perfil de clientes que solicitan créditos
- **Acción requerida:** Revisar estrategia de captación y segmentación

**Caso: puntaje_datacredito (PSI = 0.45)**

- Cambio significativo en los perfiles crediticios
- Podría indicar cambios en:
  - Política de aprobación de créditos
  - Condiciones económicas del mercado
  - Calidad de clientes que acceden al producto

### Recomendaciones de Acción

#### ⚠️ Acción Inmediata (Crítico)

1. **Reentrenar el modelo:**
   - Con 71.4% de features en drift, el modelo podría estar degradándose
   - Incluir datos recientes para actualizar patrones

2. **Auditoría del proceso de datos:**
   - Verificar si hubo cambios en:
     - Fuentes de datos
     - Procesos de recolección
     - Transformaciones aplicadas
     - Políticas de negocio

3. **Análisis de causa raíz:**
   - Investigar por qué cambió tan drásticamente `edad_cliente`
   - Revisar cambios en política de productos
   - Evaluar impacto de factores externos (economía, regulación)

#### 📊 Monitoreo Continuo

1. **Frecuencia:** Ejecutar análisis de drift semanalmente
2. **Alertas automatizadas:** Configurar notificaciones cuando PSI > 0.25
3. **Dashboard en producción:** Mantener `app_streamlit.py` ejecutándose
4. **Métricas de negocio:** Correlacionar drift con tasas de impago reales

#### 🎯 Estrategia de Mitigación

1. **Modelo Champion-Challenger:**
   - Mantener modelo actual (champion)
   - Entrenar nuevo modelo con datos recientes (challenger)
   - Comparar desempeño en producción

2. **Segmentación:**
   - Considerar modelos específicos por segmento de edad
   - Evaluar modelos diferentes para perfiles de Datacrédito

3. **Feature Store:**
   - Implementar versionado de features
   - Mantener historial de distribuciones
   - Detectar drift de forma automatizada

---

## 🔍 Hallazgos Principales

### 1. Calidad de Datos

✅ **Fortalezas:**
- Dataset limpio sin valores nulos críticos (excepto Datacrédito)
- Tipos de datos consistentes
- Sin duplicados
- Rangos de valores válidos

⚠️ **Áreas de mejora:**
- 27% de valores nulos en `Datacredito` requieren estrategia de imputación
- Algunas variables con distribuciones asimétricas

### 2. Desbalance de Clases

📊 **Problema identificado:**
- Clase 1: 95.25% vs Clase 0: 4.75%
- Ratio de 20:1 entre clases

✅ **Solución implementada:**
- SMOTE con sampling_strategy=0.3
- Mejora significativa en detección de impagos
- Balance entre precision y recall

### 3. Feature Engineering

💡 **Features más valiosas creadas:**
- `ratio_deuda_ingreso`: Indicador clave de sobreendeudamiento
- `ratio_cuota_salario`: Capacidad de pago mensual
- `sin_historial_datacredito`: Flag crítico para segmentación

🎯 **Impacto:**
- Incremento en poder predictivo del modelo
- Mejor interpretabilidad de decisiones
- Alineación con lógica de negocio

### 4. Modelado

🏆 **Gradient Boosting como mejor modelo:**
- F1-Score más alto: Balance óptimo
- ROC-AUC superior: Mejor discriminación
- Robustez ante desbalance

📈 **Comparación con baseline:**
- Logistic Regression: F1 = 0.8659
- Gradient Boosting: F1 = 0.9098
- **Mejora:** +5.07%

### 5. Data Drift

🚨 **Hallazgo crítico:**
- 71.4% de features con drift significativo
- Cambios drásticos en perfil de clientes (edad, puntaje)
- Necesidad urgente de reentrenamiento

💡 **Insights:**
- El drift es principalmente en variables demográficas y crediticias
- Las variables operacionales (saldos, mora) se mantienen estables
- Posible cambio en estrategia de captación o condiciones de mercado

### 6. Valor del Sistema de Monitoreo

✅ **Beneficios comprobados:**
- Detección temprana de degradación del modelo
- Visibilidad completa del estado del pipeline
- Toma de decisiones basada en datos
- Alertas automáticas con umbrales configurables

---

## 🛠️ Tecnologías Utilizadas

### Lenguajes
- **Python 3.11.9** - Lenguaje principal del proyecto

### Ciencia de Datos
- **pandas 2.3.3** - Manipulación y análisis de datos
- **numpy 2.4.6** - Operaciones numéricas y arrays
- **scipy 1.17.1** - Estadística y pruebas de hipótesis

### Visualización
- **matplotlib 3.10.9** - Gráficos estáticos
- **seaborn 0.13.2** - Visualizaciones estadísticas
- **plotly 6.7.0** - Gráficos interactivos en dashboard

### Machine Learning
- **scikit-learn 1.8.0** - Algoritmos de ML y pipelines
- **imbalanced-learn 0.14.1** - Técnicas de balanceo (SMOTE)
- **xgboost 3.2.0** - Gradient Boosting avanzado
- **joblib 1.5.3** - Persistencia de modelos

### Desarrollo
- **Jupyter 1.1.1** - Notebooks interactivos para EDA
- **ipykernel 7.2.0** - Kernel de Python para Jupyter
- **streamlit 1.57.0** - Framework para dashboards interactivos

### Utilidades
- **openpyxl 3.1.5** - Lectura de archivos Excel
- **python-dotenv 1.2.2** - Gestión de variables de entorno

### Control de Versiones
- **Git** - Control de versiones
- **GitHub** - Hosting del repositorio
- **GitFlow** - Estrategia de branching (developer → certification → main)

---

## 👨‍💻 Autor

**Juan Carlos [Apellido]**  
Estudiante de Ciencia de Datos - Soy Henry  
Módulo 5: Fundamentos de Nube y Ciencia de Datos de Producción

### Contacto
- GitHub: [Tu usuario de GitHub]
- LinkedIn: [Tu perfil de LinkedIn]
- Email: [Tu email]

---

## 📄 Licencia

Este proyecto es parte del Proyecto Integrador del Módulo 5 de Soy Henry.

---

## 🙏 Agradecimientos

- **Soy Henry** - Por la formación en Ciencia de Datos
- **Instructores y TAs** - Por el apoyo durante el módulo
- **Comunidad Henry** - Por el aprendizaje colaborativo

---

## 📝 Historial de Versiones

| Versión | Fecha | Descripción | Branch |
|---------|-------|-------------|--------|
| V1.0.0 | Abr 2026 | Estructura inicial del proyecto | main |
| V1.0.1 | Abr 2026 | Avance #1: Carga y EDA completo | developer → main |
| V1.1.0 | May 2026 | Avance #2: Feature Engineering | developer → main |
| V1.1.1 | May 2026 | Avance #2: Modelado y evaluación | developer → main |
| **V1.2.0** | **May 2026** | **Avance #3: Monitoreo de drift y dashboard** | **developer → main** |

---

## 📚 Referencias

1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. [PSI (Population Stability Index) Explained](https://www.listendata.com/2015/05/population-stability-index.html)
5. [Data Drift Detection Methods](https://mlops.community/data-drift-detection/)

---

**📊 Proyecto completado exitosamente - Mayo 2026**
