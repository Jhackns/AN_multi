import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import os
import pickle

from neural_network import NeuralNetwork

def load_diabetes_data(path: str, names: list[str] | None = None) -> pd.DataFrame:
    """Carga los datos de diabetes desde un archivo CSV."""
    df = pd.read_csv(path, sep=",", header=None if names is not None else 'infer', names=names)
    return df

def replace_zero_with_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reemplaza valores cero con la media en columnas específicas."""
    df = df.copy()
    for col in columns:
        vals = df[col].values.astype(float)
        # Compute mean excluding zeros
        non_zero_vals = vals[vals != 0]
        mean_val = non_zero_vals.mean() if len(non_zero_vals) > 0 else 0.0
        vals[vals == 0] = mean_val
        df[col] = vals
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Preprocesa los datos de diabetes."""
    # Replace zeros in selected columns
    invalid_zero_cols = [
        "Glucose",
        "BloodPressure", 
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    df = replace_zero_with_mean(df, invalid_zero_cols)
    # Separate features and target
    X_df = df.drop(columns=["Outcome"])
    y = df[["Outcome"]].values.astype(int)
    # Standardise all features
    mean = X_df.mean(axis=0)
    std = X_df.std(axis=0).replace(0, 1)
    X_norm = (X_df - mean) / std
    return X_norm.values.astype(float), y.astype(float), {"mean": mean.values, "std": std.values, "columns": X_df.columns.tolist()}

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    n_test = int(len(indices) * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calcula la matriz de confusión."""
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])

def create_visualizations(history: Dict, y_test: np.ndarray, y_pred_probs: np.ndarray, 
                         y_pred_labels: np.ndarray, train_metrics: Dict, test_metrics: Dict,
                         output_dir: str = "resultados_diabetes") -> None:
    """Crea visualizaciones mejoradas para el modelo de diabetes."""
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar estilo de matplotlib
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Curva de pérdida mejorada
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], linewidth=2, color='#2E86AB')
    plt.xlabel("Época", fontsize=12)
    plt.ylabel("Pérdida (Binary Crossentropy)", fontsize=12)
    plt.title("Evolución de la Pérdida Durante el Entrenamiento", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Distribución de probabilidades predichas
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_probs[y_test.flatten() == 0], bins=30, alpha=0.7, label='Sin Diabetes', color='#A23B72')
    plt.hist(y_pred_probs[y_test.flatten() == 1], bins=30, alpha=0.7, label='Con Diabetes', color='#F18F01')
    plt.xlabel("Probabilidad Predicha", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title("Distribución de Probabilidades Predichas", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diabetes_analisis_entrenamiento.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Matriz de confusión mejorada
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(8, 6))
    
    # Calcular porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear anotaciones combinadas
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', cbar_kws={'label': 'Número de casos'},
                xticklabels=['Predicción: Sin Diabetes', 'Predicción: Con Diabetes'],
                yticklabels=['Real: Sin Diabetes', 'Real: Con Diabetes'])
    plt.xlabel("Predicción del Modelo", fontsize=12)
    plt.ylabel("Diagnóstico Real", fontsize=12)
    plt.title("Matriz de Confusión - Diagnóstico de Diabetes", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diabetes_matriz_confusion.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Métricas comparativas
    metrics_names = ['Accuracy', 'Precisión', 'Recall', 'F1-Score']
    train_values = [train_metrics['accuracy'], train_metrics['precision'], 
                   train_metrics['recall'], train_metrics['f1']]
    test_values = [test_metrics['accuracy'], test_metrics['precision'], 
                  test_metrics['recall'], test_metrics['f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_values, width, label='Entrenamiento', color='#2E86AB', alpha=0.8)
    plt.bar(x + width/2, test_values, width, label='Prueba', color='#F18F01', alpha=0.8)
    
    plt.xlabel('Métricas', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.title('Comparación de Métricas: Entrenamiento vs Prueba', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)
    
    # Añadir valores en las barras
    for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
        plt.text(i - width/2, train_val + 0.01, f'{train_val:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, test_val + 0.01, f'{test_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diabetes_metricas_comparacion.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_model_and_stats(model: NeuralNetwork, stats: Dict, output_dir: str = "resultados_diabetes") -> None:
    """Guarda el modelo entrenado y las estadísticas de normalización."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar estadísticas de normalización
    with open(os.path.join(output_dir, "diabetes_normalization_stats.pkl"), 'wb') as f:
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
    
    with open(os.path.join(output_dir, "diabetes_model.pkl"), 'wb') as f:
        pickle.dump(model_info, f)

def load_model_and_stats(output_dir: str = "resultados_diabetes") -> Tuple[NeuralNetwork, Dict]:
    """Carga el modelo entrenado y las estadísticas de normalización."""
    # Cargar estadísticas
    with open(os.path.join(output_dir, "diabetes_normalization_stats.pkl"), 'rb') as f:
        stats = pickle.load(f)
    
    # Cargar modelo
    with open(os.path.join(output_dir, "diabetes_model.pkl"), 'rb') as f:
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

def predict_diabetes_interactive(model: NeuralNetwork, stats: Dict) -> None:
    """Interfaz interactiva para hacer predicciones de diabetes."""
    print("\n" + "="*60)
    print("🏥 SISTEMA DE DIAGNÓSTICO DE DIABETES")
    print("="*60)
    print("Ingrese los siguientes datos del paciente:")
    print("-"*40)
    
    feature_names = [
        "Número de embarazos",
        "Concentración de glucosa (mg/dL)",
        "Presión arterial diastólica (mm Hg)",
        "Grosor del pliegue cutáneo del tríceps (mm)",
        "Insulina sérica (mu U/ml)",
        "Índice de masa corporal (kg/m²)",
        "Función de pedigrí de diabetes",
        "Edad (años)"
    ]
    
    feature_ranges = [
        "0-17",
        "0-200",
        "0-122",
        "0-99",
        "0-846",
        "0.0-67.1",
        "0.078-2.42",
        "21-81"
    ]
    
    user_input = []
    
    for i, (name, range_val) in enumerate(zip(feature_names, feature_ranges)):
        while True:
            try:
                value = float(input(f"{i+1}. {name} (rango típico: {range_val}): "))
                user_input.append(value)
                break
            except ValueError:
                print("   ❌ Por favor, ingrese un número válido.")
    
    # Normalizar entrada del usuario
    user_array = np.array(user_input).reshape(1, -1)
    user_normalized = (user_array - stats['mean']) / stats['std']
    
    # Hacer predicción
    probability = model.predict(user_normalized)[0, 0]
    prediction = 1 if probability >= 0.5 else 0
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("📊 RESULTADOS DEL DIAGNÓSTICO")
    print("="*60)
    
    if prediction == 1:
        print(f"🔴 RIESGO ALTO DE DIABETES")
        print(f"   Probabilidad: {probability:.1%}")
        print(f"   Recomendación: Consulte con un médico especialista")
    else:
        print(f"🟢 RIESGO BAJO DE DIABETES")
        print(f"   Probabilidad: {probability:.1%}")
        print(f"   Recomendación: Mantenga hábitos saludables")
    
    print("\n📋 INTERPRETACIÓN:")
    print(f"   • Confianza del modelo: {max(probability, 1-probability):.1%}")
    
    if probability > 0.7:
        print("   • Riesgo muy alto - Consulta médica urgente recomendada")
    elif probability > 0.5:
        print("   • Riesgo moderado-alto - Monitoreo médico recomendado")
    elif probability > 0.3:
        print("   • Riesgo bajo-moderado - Revisión médica preventiva")
    else:
        print("   • Riesgo muy bajo - Continúe con hábitos saludables")
    
    print("\n⚠️  NOTA IMPORTANTE:")
    print("   Este sistema es una herramienta de apoyo y NO reemplaza")
    print("   el diagnóstico médico profesional.")
    print("="*60)

def main() -> None:
    """Función principal del sistema de diagnóstico de diabetes."""
    print("🚀 Iniciando Sistema de Diagnóstico de Diabetes con Redes Neuronales")
    print("="*70)
    
    # Configurar rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "diabetes.csv")
    output_dir = "resultados_diabetes"
    
    # Nombres de columnas para el dataset de diabetes Pima Indians
    columns = [
        "Pregnancies",
        "Glucose", 
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    
    print("📊 Cargando y preprocesando datos...")
    df = load_diabetes_data(data_path, names=columns)
    X, y, stats = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   • Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"   • Datos de prueba: {X_test.shape[0]} muestras")
    print(f"   • Características: {X_train.shape[1]}")
    
    # Definir arquitectura de la red neuronal
    n_features = X_train.shape[1]
    layer_sizes = [n_features, 16, 8, 1]
    
    print(f"\n🧠 Configurando Red Neuronal:")
    print(f"   • Arquitectura: {' → '.join(map(str, layer_sizes))}")
    print(f"   • Función de activación oculta: ReLU")
    print(f"   • Función de activación salida: Sigmoid")
    print(f"   • Función de pérdida: Binary Crossentropy")
    
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_activation="relu",
        output_activation="sigmoid", 
        loss="binary_crossentropy",
        learning_rate=0.01,
        random_state=42,
    )
    
    print("\n⚡ Entrenando modelo...")
    history = nn.fit(X_train, y_train, epochs=500, verbose=True)
    
    print("\n📈 Evaluando rendimiento...")
    # Evaluar métricas
    train_metrics = nn.evaluate_classification(X_train, y_train)
    test_metrics = nn.evaluate_classification(X_test, y_test)
    
    # Hacer predicciones
    y_pred_probs = nn.predict(X_test)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    
    # Crear visualizaciones
    print("🎨 Generando visualizaciones...")
    create_visualizations(history, y_test, y_pred_probs, y_pred_labels, 
                         train_metrics, test_metrics, output_dir)
    
    # Guardar modelo
    print("💾 Guardando modelo entrenado...")
    save_model_and_stats(nn, stats, output_dir)
    
    # Mostrar tabla de resultados
    results_df = pd.DataFrame([
        ["Accuracy", train_metrics["accuracy"], test_metrics["accuracy"]],
        ["Precisión", train_metrics["precision"], test_metrics["precision"]],
        ["Recall", train_metrics["recall"], test_metrics["recall"]],
        ["F1-Score", train_metrics["f1"], test_metrics["f1"]],
    ], columns=["Métrica", "Entrenamiento", "Prueba"])
    
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("="*50)
    print(results_df.to_string(index=False, formatters={
        "Entrenamiento": lambda x: f"{x:.3f}",
        "Prueba": lambda x: f"{x:.3f}",
    }))
    
    print(f"\n📁 Archivos generados en '{output_dir}':")
    print("   • diabetes_analisis_entrenamiento.png - Análisis del entrenamiento")
    print("   • diabetes_matriz_confusion.png - Matriz de confusión")
    print("   • diabetes_metricas_comparacion.png - Comparación de métricas")
    print("   • diabetes_model.pkl - Modelo entrenado")
    print("   • diabetes_normalization_stats.pkl - Estadísticas de normalización")
    
    # Interfaz interactiva
    print("\n" + "="*70)
    while True:
        respuesta = input("¿Desea hacer una predicción interactiva? (s/n): ").lower().strip()
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            predict_diabetes_interactive(nn, stats)
            print("\n" + "="*70)
        elif respuesta in ['n', 'no']:
            break
        else:
            print("Por favor, responda 's' para sí o 'n' para no.")
    
    print("✅ Proceso completado exitosamente!")
    print("="*70)

if __name__ == "__main__":
    main()
