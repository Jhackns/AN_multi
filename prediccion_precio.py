import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import os
import pickle

from neural_network import NeuralNetwork

def load_housing_data(path: str) -> pd.DataFrame:
    """Carga los datos de viviendas de Boston desde un archivo CSV."""
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame, target_col: str = "MEDV") -> Tuple[np.ndarray, np.ndarray, dict]:
    """Preprocesa los datos de viviendas de Boston."""
    # Separate features and target
    X_df = df.drop(columns=[target_col])
    y = df[[target_col]].values.astype(float)
    
    # Identify categorical and numerical columns
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # One-hot encode categorical features
    if categorical_cols:
        X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
    else:
        X_encoded = X_df.copy()
    
    # Standardize numerical features
    mean = X_encoded.mean(axis=0)
    std = X_encoded.std(axis=0).replace(0, 1)
    X_norm = (X_encoded - mean) / std
    
    stats = {
        "mean": mean.values,
        "std": std.values,
        "columns": X_encoded.columns.tolist(),
        "target_col": target_col,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "target_mean": y.mean(),
        "target_std": y.std()
    }
    
    return X_norm.values.astype(float), y.astype(float), stats

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    n_test = int(len(indices) * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def create_visualizations(history: Dict, y_train: np.ndarray, y_test: np.ndarray, 
                         y_train_pred: np.ndarray, y_test_pred: np.ndarray,
                         train_metrics: Dict, test_metrics: Dict,
                         output_dir: str = "resultados_boston") -> None:
    """Crea visualizaciones mejoradas para el modelo de predicción de precios."""
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar estilo de matplotlib
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Curva de pérdida y métricas de entrenamiento
    plt.figure(figsize=(15, 5))
    
    # Curva de pérdida
    plt.subplot(1, 3, 1)
    plt.plot(history["loss"], linewidth=2, color='#2E86AB')
    plt.xlabel("Época", fontsize=12)
    plt.ylabel("Pérdida (MSE)", fontsize=12)
    plt.title("Evolución de la Pérdida Durante el Entrenamiento", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Distribución de residuos
    plt.subplot(1, 3, 2)
    residuals_train = y_train.flatten() - y_train_pred.flatten()
    residuals_test = y_test.flatten() - y_test_pred.flatten()
    
    plt.hist(residuals_train, bins=30, alpha=0.7, label='Entrenamiento', color='#2E86AB')
    plt.hist(residuals_test, bins=30, alpha=0.7, label='Prueba', color='#F18F01')
    plt.xlabel("Residuos (Real - Predicho)", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title("Distribución de Residuos", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot para normalidad de residuos
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(residuals_test, dist="norm", plot=plt)
    plt.title("Q-Q Plot - Normalidad de Residuos", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boston_analisis_entrenamiento.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Predicciones vs Valores Reales (mejorado)
    plt.figure(figsize=(12, 5))
    
    # Conjunto de entrenamiento
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='#2E86AB', s=30)
    
    # Línea de predicción perfecta
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    plt.xlabel("Precio Real ($1000s)", fontsize=12)
    plt.ylabel("Precio Predicho ($1000s)", fontsize=12)
    plt.title(f"Entrenamiento\nR² = {train_metrics['r2']:.3f}, RMSE = {train_metrics['rmse']:.2f}", 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Conjunto de prueba
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='#F18F01', s=30)
    
    # Línea de predicción perfecta
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    plt.xlabel("Precio Real ($1000s)", fontsize=12)
    plt.ylabel("Precio Predicho ($1000s)", fontsize=12)
    plt.title(f"Prueba\nR² = {test_metrics['r2']:.3f}, RMSE = {test_metrics['rmse']:.2f}", 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boston_predicciones_vs_reales.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Análisis de errores
    plt.figure(figsize=(12, 8))
    
    # Error absoluto vs precio real
    plt.subplot(2, 2, 1)
    abs_errors = np.abs(residuals_test)
    plt.scatter(y_test.flatten(), abs_errors, alpha=0.6, color='#A23B72')
    plt.xlabel("Precio Real ($1000s)", fontsize=12)
    plt.ylabel("Error Absoluto", fontsize=12)
    plt.title("Error Absoluto vs Precio Real", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Error relativo
    plt.subplot(2, 2, 2)
    relative_errors = np.abs(residuals_test) / y_test.flatten() * 100
    plt.hist(relative_errors, bins=30, alpha=0.7, color='#F18F01')
    plt.xlabel("Error Relativo (%)", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title("Distribución de Errores Relativos", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Métricas comparativas
    plt.subplot(2, 2, 3)
    metrics_names = ['RMSE', 'R²']
    train_values = [train_metrics['rmse'], train_metrics['r2']]
    test_values = [test_metrics['rmse'], test_metrics['r2']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, train_values, width, label='Entrenamiento', color='#2E86AB', alpha=0.8)
    plt.bar(x + width/2, test_values, width, label='Prueba', color='#F18F01', alpha=0.8)
    
    plt.xlabel('Métricas', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.title('Comparación de Métricas', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
        plt.text(i - width/2, train_val + max(train_values) * 0.01, f'{train_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
        plt.text(i + width/2, test_val + max(test_values) * 0.01, f'{test_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Estadísticas de error
    plt.subplot(2, 2, 4)
    error_stats = [
        f"Error medio: {np.mean(residuals_test):.2f}",
        f"Error mediano: {np.median(residuals_test):.2f}",
        f"Error std: {np.std(residuals_test):.2f}",
        f"Error máximo: {np.max(np.abs(residuals_test)):.2f}",
        f"Error relativo medio: {np.mean(relative_errors):.1f}%"
    ]
    
    plt.text(0.1, 0.9, "Estadísticas de Error:", fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)
    
    for i, stat in enumerate(error_stats):
        plt.text(0.1, 0.7 - i*0.12, stat, fontsize=12, transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boston_analisis_errores.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_model_and_stats(model: NeuralNetwork, stats: Dict, output_dir: str = "resultados_boston") -> None:
    """Guarda el modelo entrenado y las estadísticas de normalización."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar estadísticas de normalización
    with open(os.path.join(output_dir, "boston_normalization_stats.pkl"), 'wb') as f:
        pickle.dump(stats, f)
    
    # Guardar información del modelo
    model_info = {
        'layer_sizes': model.layer_sizes,
        'weights': model.weights,
        'biases': model.biases,
        'hidden_activation': model.hidden_activation,
        'output_activation': model.output_activation,
        'loss': model.loss
    }
    
    with open(os.path.join(output_dir, "boston_model.pkl"), 'wb') as f:
        pickle.dump(model_info, f)

def load_model_and_stats(output_dir: str = "resultados_boston") -> Tuple[NeuralNetwork, Dict]:
    """Carga el modelo entrenado y las estadísticas de normalización."""
    # Cargar estadísticas
    with open(os.path.join(output_dir, "boston_normalization_stats.pkl"), 'rb') as f:
        stats = pickle.load(f)
    
    # Cargar modelo
    with open(os.path.join(output_dir, "boston_model.pkl"), 'rb') as f:
        model_info = pickle.load(f)
    
    # Recrear modelo
    model = NeuralNetwork(
        layer_sizes=model_info['layer_sizes'],
        hidden_activation=model_info['hidden_activation'],
        output_activation=model_info['output_activation'],
        loss=model_info['loss']
    )
    model.weights = model_info['weights']
    model.biases = model_info['biases']
    
    return model, stats

def predict_price_interactive(model: NeuralNetwork, stats: Dict) -> None:
    """Interfaz interactiva para hacer predicciones de precios de viviendas."""
    print("\n" + "="*60)
    print("🏠 SISTEMA DE PREDICCIÓN DE PRECIOS DE VIVIENDAS")
    print("="*60)
    print("Ingrese las siguientes características de la vivienda:")
    print("-"*50)
    
    feature_descriptions = {
        "CRIM": ("Tasa de criminalidad per cápita por ciudad", "0.0-89.0"),
        "ZN": ("Proporción de terreno residencial zonificado para lotes > 25,000 sq.ft", "0.0-100.0"),
        "INDUS": ("Proporción de acres comerciales no minoristas por ciudad", "0.0-27.7"),
        "CHAS": ("Variable ficticia del río Charles (1 si limita con río, 0 si no)", "0 o 1"),
        "NOX": ("Concentración de óxidos nítricos (partes por 10 millones)", "0.38-0.87"),
        "RM": ("Número promedio de habitaciones por vivienda", "3.6-8.8"),
        "AGE": ("Proporción de unidades ocupadas por propietarios construidas antes de 1940", "2.9-100.0"),
        "DIS": ("Distancias ponderadas a cinco centros de empleo de Boston", "1.1-12.1"),
        "RAD": ("Índice de accesibilidad a autopistas radiales", "1-24"),
        "TAX": ("Tasa de impuesto a la propiedad de valor completo por $10,000", "187-711"),
        "PTRATIO": ("Relación alumno-maestro por ciudad", "12.6-22.0"),
        "LSTAT": ("% de estatus más bajo de la población", "1.7-37.0")
    }
    
    user_input = []
    
    for i, (feature, (description, range_val)) in enumerate(feature_descriptions.items()):
        print(f"\n{i+1}. {feature}: {description}")
        print(f"   Rango típico: {range_val}")
        
        while True:
            try:
                value = float(input(f"   Ingrese valor: "))
                user_input.append(value)
                break
            except ValueError:
                print("   ❌ Por favor, ingrese un número válido.")
    
    # Normalizar entrada del usuario
    user_array = np.array(user_input).reshape(1, -1)
    user_normalized = (user_array - stats['mean']) / stats['std']
    
    # Hacer predicción
    predicted_price = model.predict(user_normalized)[0, 0]
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("💰 PREDICCIÓN DE PRECIO")
    print("="*60)
    
    print(f"🏠 Precio estimado: ${predicted_price:.2f}k")
    print(f"   (${predicted_price * 1000:.0f} USD)")
    
    # Contexto del mercado
    target_mean = stats.get('target_mean', 22.5)
    target_std = stats.get('target_std', 9.2)
    
    if predicted_price > target_mean + target_std:
        categoria = "🔴 PRECIO ALTO"
        descripcion = "Por encima del promedio del mercado"
    elif predicted_price > target_mean:
        categoria = "🟡 PRECIO MEDIO-ALTO"
        descripcion = "Ligeramente por encima del promedio"
    elif predicted_price > target_mean - target_std:
        categoria = "🟢 PRECIO MEDIO-BAJO"
        descripcion = "Ligeramente por debajo del promedio"
    else:
        categoria = "🔵 PRECIO BAJO"
        descripcion = "Por debajo del promedio del mercado"
    
    print(f"\n📊 ANÁLISIS DE MERCADO:")
    print(f"   • Categoría: {categoria}")
    print(f"   • Descripción: {descripcion}")
    print(f"   • Precio promedio del mercado: ${target_mean:.2f}k")
    
    # Rango de confianza (estimación simple)
    confidence_interval = predicted_price * 0.15  # ±15% como estimación
    print(f"\n📈 RANGO ESTIMADO:")
    print(f"   • Precio mínimo: ${predicted_price - confidence_interval:.2f}k")
    print(f"   • Precio máximo: ${predicted_price + confidence_interval:.2f}k")
    
    print("\n⚠️  NOTA IMPORTANTE:")
    print("   Esta predicción es una estimación basada en datos históricos")
    print("   y no constituye una valoración profesional inmobiliaria.")
    print("="*60)

def main() -> None:
    """Función principal del sistema de predicción de precios de viviendas."""
    print("🚀 Iniciando Sistema de Predicción de Precios de Viviendas con Redes Neuronales")
    print("="*80)
    
    # Configurar rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "Boston.csv")
    output_dir = "resultados_boston"
    
    print("📊 Cargando y preprocesando datos...")
    df = load_housing_data(data_path)
    X, y, stats = preprocess_data(df, target_col="MEDV")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   • Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"   • Datos de prueba: {X_test.shape[0]} muestras")
    print(f"   • Características: {X_train.shape[1]}")
    print(f"   • Precio promedio: ${y.mean():.2f}k (±${y.std():.2f}k)")
    
    # Definir arquitectura de la red neuronal
    n_features = X_train.shape[1]
    layer_sizes = [n_features, 32, 16, 1]
    
    print(f"\n🧠 Configurando Red Neuronal:")
    print(f"   • Arquitectura: {' → '.join(map(str, layer_sizes))}")
    print(f"   • Función de activación oculta: ReLU")
    print(f"   • Función de activación salida: Linear")
    print(f"   • Función de pérdida: MSE")
    
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_activation="relu",
        output_activation="linear",
        loss="mse",
        learning_rate=0.001,
        random_state=42,
    )
    
    print("\n⚡ Entrenando modelo...")
    history = nn.fit(X_train, y_train, epochs=800, verbose=True)
    
    print("\n📈 Evaluando rendimiento...")
    # Evaluar métricas
    train_metrics = nn.evaluate_regression(X_train, y_train)
    test_metrics = nn.evaluate_regression(X_test, y_test)
    
    # Hacer predicciones
    y_train_pred = nn.predict(X_train)
    y_test_pred = nn.predict(X_test)
    
    # Crear visualizaciones
    print("🎨 Generando visualizaciones...")
    create_visualizations(history, y_train, y_test, y_train_pred, y_test_pred,
                         train_metrics, test_metrics, output_dir)
    
    # Guardar modelo
    print("💾 Guardando modelo entrenado...")
    save_model_and_stats(nn, stats, output_dir)
    
    # Mostrar tabla de resultados
    results_df = pd.DataFrame([
        ["RMSE", train_metrics["rmse"], test_metrics["rmse"]],
        ["R²", train_metrics["r2"], test_metrics["r2"]],
    ], columns=["Métrica", "Entrenamiento", "Prueba"])
    
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("="*50)
    print(results_df.to_string(index=False, formatters={
        "Entrenamiento": lambda x: f"{x:.3f}",
        "Prueba": lambda x: f"{x:.3f}",
    }))
    
    print(f"\n📁 Archivos generados en '{output_dir}':")
    print("   • boston_analisis_entrenamiento.png - Análisis del entrenamiento")
    print("   • boston_predicciones_vs_reales.png - Predicciones vs valores reales")
    print("   • boston_analisis_errores.png - Análisis detallado de errores")
    print("   • boston_model.pkl - Modelo entrenado")
    print("   • boston_normalization_stats.pkl - Estadísticas de normalización")
    
    # Interfaz interactiva
    print("\n" + "="*80)
    while True:
        respuesta = input("¿Desea hacer una predicción interactiva? (s/n): ").lower().strip()
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            predict_price_interactive(nn, stats)
            print("\n" + "="*80)
        elif respuesta in ['n', 'no']:
            break
        else:
            print("Por favor, responda 's' para sí o 'n' para no.")
    
    print("✅ Proceso completado exitosamente!")
    print("="*80)

if __name__ == "__main__":
    main()
