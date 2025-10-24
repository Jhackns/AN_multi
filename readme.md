# Guía Didáctica del Sistema de Redes Neuronales

## Propósito
Este proyecto implementa redes neuronales multicapa (MLP) desde cero para dos objetivos educativos:
- Diagnóstico de diabetes (clasificación binaria)
- Predicción de precios de viviendas (regresión)

## Cómo Funciona la Red Neuronal
- Arquitectura: capas de entrada, varias capas ocultas y capa de salida.
- Activaciones: `ReLU` en ocultas; `Sigmoid` en salida para clasificación; `Linear` en salida para regresión.
- Pérdidas: `Binary Crossentropy` (clasificación) y `MSE` (regresión).
- Entrenamiento: propagación hacia adelante, cálculo de pérdida y retropropagación de errores (backpropagation) para actualizar pesos.
- Evaluación: métricas de clasificación (accuracy, precision, recall, F1) y regresión (RMSE, R²).
- Resultados: gráficos y modelos guardados en carpetas `resultados_diabetes/` y `resultados_boston/`.

## Estructura del Proyecto
```
redes-neuronales/
├── main.py                 # Punto de entrada unificado (ejecución recomendada)
├── neural_network.py       # Implementación de la MLP y utilidades
├── diagnostico_diabetes.py # Módulo de clasificación (diabetes)
├── prediccion_precio.py    # Módulo de regresión (Boston)
├── data/                   # Datasets CSV
├── resultados_diabetes/    # Salidas del sistema de diabetes
├── resultados_boston/      # Salidas del sistema de precios
├── requirements.txt        # Dependencias
└── GUIA_EJECUCION.md       # Guía ampliada de uso
```

## Dependencias
Instale las librerías principales:
```bash
pip install -r requirements.txt
```
Incluye: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`.

## Advertencia Importante
Para ejecutar el programa y entrenar los modelos, **solo debe** ejecutar `main.py`. No es necesario ni recomendable ejecutar directamente otros archivos.

## Ejecución Paso a Paso
1. Abra una consola en la carpeta del proyecto.
2. (Opcional) Cree y active un entorno virtual.
3. Instale dependencias: `pip install -r requirements.txt`.
4. Verifique que los archivos `data/diabetes.csv` y `data/Boston.csv` existan.
5. Ejecute el sistema:
   ```bash
   python main.py
   ```
6. Elija el módulo (diabetes o precios) desde el menú.
7. Espere el entrenamiento automático.
8. Ingrese datos en la interfaz interactiva para obtener predicciones.
9. Revise métricas y gráficos generados en `resultados_diabetes/` o `resultados_boston/`.

## Resultados Esperados
- Clasificación (diabetes): métricas, curva de pérdida, matriz de confusión, modelo `.pkl`.
- Regresión (Boston): métricas, gráficos de residuos y predicción vs real, modelo `.pkl`.

## Referencia
Fundamentos teóricos y ejemplos: https://cienciadedatos.net/documentos/py35-redes-neuronales-python