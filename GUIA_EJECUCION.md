# 🧠 Guía de Ejecución - Sistema de Redes Neuronales

## 📋 Descripción General

Este sistema implementa redes neuronales desde cero para dos casos de uso principales:

1. **🏥 Diagnóstico de Diabetes** (Clasificación binaria)
2. **🏠 Predicción de Precios de Viviendas** (Regresión)

## 🚀 Instalación y Configuración

### 1. Requisitos del Sistema

- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, macOS, Linux
- **RAM**: Mínimo 4GB recomendado
- **Espacio en disco**: ~100MB para datos y resultados

### 2. Instalación de Dependencias

```bash
# Opción 1: Instalar desde requirements.txt
pip install -r requirements.txt

# Opción 2: Instalar manualmente
pip install numpy pandas matplotlib seaborn scipy
```

### 3. Verificación de Archivos

Asegúrese de tener la siguiente estructura de archivos:

```
redes-neuronales/
├── main.py                    # Script principal
├── neural_network.py          # Implementación de la red neuronal
├── diagnostico_diabetes.py    # Sistema de diagnóstico de diabetes
├── prediccion_precio.py       # Sistema de predicción de precios
├── requirements.txt           # Dependencias
├── GUIA_EJECUCION.md         # Esta guía
└── data/
    ├── diabetes.csv          # Dataset de diabetes
    └── Boston.csv            # Dataset de viviendas de Boston
```

## 🎯 Formas de Ejecución

### Método 1: Script Principal Unificado (RECOMENDADO)

```bash
python main.py
```

Este método ofrece:
- ✅ Menú interactivo
- ✅ Verificación automática de dependencias
- ✅ Selección entre ambos sistemas
- ✅ Manejo de errores

### Método 2: Ejecución Individual

#### Para Diagnóstico de Diabetes:
```bash
python diagnostico_diabetes.py
```

#### Para Predicción de Precios:
```bash
python prediccion_precio.py
```

## 📊 Qué Esperar de Cada Sistema

### 🏥 Sistema de Diagnóstico de Diabetes

#### Proceso de Entrenamiento:
1. **Carga de datos**: 768 muestras del dataset Pima Indians
2. **Preprocesamiento**: 
   - Reemplazo de valores cero con medias
   - Normalización de características
3. **Entrenamiento**: 500 épocas con arquitectura 8→16→8→1
4. **Evaluación**: Métricas de clasificación

#### Salidas Generadas:
- **Carpeta**: `resultados_diabetes/`
- **Archivos**:
  - `diabetes_analisis_entrenamiento.png` - Curva de pérdida y distribución de probabilidades
  - `diabetes_matriz_confusion.png` - Matriz de confusión detallada
  - `diabetes_metricas_comparacion.png` - Comparación de métricas
  - `diabetes_model.pkl` - Modelo entrenado
  - `diabetes_normalization_stats.pkl` - Estadísticas de normalización

#### Métricas Esperadas:
- **Accuracy**: ~75-80%
- **Precision**: ~70-75%
- **Recall**: ~60-70%
- **F1-Score**: ~65-72%

#### Interfaz Interactiva:
El sistema solicitará 8 parámetros médicos:
1. Número de embarazos (0-17)
2. Concentración de glucosa (0-200 mg/dL)
3. Presión arterial diastólica (0-122 mm Hg)
4. Grosor del pliegue cutáneo (0-99 mm)
5. Insulina sérica (0-846 mu U/ml)
6. Índice de masa corporal (0.0-67.1 kg/m²)
7. Función de pedigrí de diabetes (0.078-2.42)
8. Edad (21-81 años)

**Ejemplo de entrada**:
```
1. Número de embarazos: 2
2. Concentración de glucosa: 120
3. Presión arterial diastólica: 80
4. Grosor del pliegue cutáneo: 25
5. Insulina sérica: 100
6. Índice de masa corporal: 28.5
7. Función de pedigrí de diabetes: 0.5
8. Edad: 35
```

### 🏠 Sistema de Predicción de Precios de Viviendas

#### Proceso de Entrenamiento:
1. **Carga de datos**: 506 muestras del dataset Boston Housing
2. **Preprocesamiento**: 
   - Normalización de características numéricas
   - Codificación one-hot para categóricas
3. **Entrenamiento**: 800 épocas con arquitectura 13→32→16→1
4. **Evaluación**: Métricas de regresión

#### Salidas Generadas:
- **Carpeta**: `resultados_boston/`
- **Archivos**:
  - `boston_analisis_entrenamiento.png` - Análisis del entrenamiento y residuos
  - `boston_predicciones_vs_reales.png` - Scatter plots de predicciones
  - `boston_analisis_errores.png` - Análisis detallado de errores
  - `boston_model.pkl` - Modelo entrenado
  - `boston_normalization_stats.pkl` - Estadísticas de normalización

#### Métricas Esperadas:
- **RMSE**: ~4-6 (miles de dólares)
- **R²**: ~0.65-0.75

#### Interfaz Interactiva:
El sistema solicitará 12 características de la vivienda:
1. CRIM: Tasa de criminalidad (0.0-89.0)
2. ZN: Proporción de terreno residencial (0.0-100.0)
3. INDUS: Proporción de acres comerciales (0.0-27.7)
4. CHAS: Variable del río Charles (0 o 1)
5. NOX: Concentración de óxidos nítricos (0.38-0.87)
6. RM: Número promedio de habitaciones (3.6-8.8)
7. AGE: Proporción de unidades antiguas (2.9-100.0)
8. DIS: Distancia a centros de empleo (1.1-12.1)
9. RAD: Índice de accesibilidad (1-24)
10. TAX: Tasa de impuesto (187-711)
11. PTRATIO: Relación alumno-maestro (12.6-22.0)
12. LSTAT: % de estatus bajo (1.7-37.0)

**Ejemplo de entrada**:
```
1. CRIM: 0.1
2. ZN: 20.0
3. INDUS: 5.0
4. CHAS: 0
5. NOX: 0.5
6. RM: 6.5
7. AGE: 50.0
8. DIS: 4.0
9. RAD: 5
10. TAX: 300
11. PTRATIO: 15.0
12. LSTAT: 10.0
```

## 🔧 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
# Instalar dependencias faltantes
pip install numpy pandas matplotlib seaborn scipy
```

### Error: "FileNotFoundError"
- Verificar que los archivos CSV estén en la carpeta `data/`
- Verificar que todos los archivos Python estén presentes

### Error: "Memory Error"
- Reducir el número de épocas en el entrenamiento
- Cerrar otras aplicaciones para liberar RAM

### Rendimiento Lento
- Considerar instalar `numba` para acelerar cálculos:
```bash
pip install numba
```

## 📈 Interpretación de Resultados

### Para Diabetes:
- **Probabilidad > 70%**: Riesgo muy alto - Consulta médica urgente
- **Probabilidad 50-70%**: Riesgo moderado-alto - Monitoreo médico
- **Probabilidad 30-50%**: Riesgo bajo-moderado - Revisión preventiva
- **Probabilidad < 30%**: Riesgo muy bajo - Hábitos saludables

### Para Precios de Viviendas:
- **Precio estimado**: En miles de dólares (multiplicar por 1000 para USD)
- **Rango de confianza**: ±15% del precio estimado
- **Categorías**: Alto, Medio-Alto, Medio-Bajo, Bajo (basado en promedio del mercado)

## ⚠️ Limitaciones y Consideraciones

1. **Datos históricos**: Los modelos se basan en datos de los años 70-80
2. **Uso educativo**: No reemplaza diagnóstico médico o valoración inmobiliaria profesional
3. **Generalización**: Los modelos pueden no funcionar bien con datos muy diferentes a los de entrenamiento
4. **Precisión**: Los resultados son estimaciones con margen de error

## 🔗 Referencias

- **Fundamentos teóricos**: https://cienciadedatos.net/documentos/py35-redes-neuronales-python
- **Dataset Diabetes**: Pima Indians Diabetes Database
- **Dataset Boston**: Boston Housing Dataset (UCI Machine Learning Repository)

## 📞 Soporte

Si encuentra problemas:
1. Verificar que todas las dependencias estén instaladas
2. Comprobar la estructura de archivos
3. Revisar los mensajes de error en la consola
4. Consultar esta guía para soluciones comunes

---

**¡Disfrute explorando el mundo de las redes neuronales!** 🚀