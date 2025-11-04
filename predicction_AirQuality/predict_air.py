import os
import time
import json
import random
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    raise SystemExit("TensorFlow/Keras no está instalado. Instale con: pip install tensorflow")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_output_dirs(base_dir: str) -> Dict[str, str]:
    """Crea carpetas de salida necesarias y devuelve rutas."""
    graficos_dir = os.path.join(base_dir, "graficos")
    os.makedirs(graficos_dir, exist_ok=True)
    return {
        "graficos": graficos_dir,
    }


def load_and_clean_data(data_path: str, station: str = "Aotizhongxin") -> pd.DataFrame:
    """Carga el CSV, combina fecha/hora a datetime, filtra estación y realiza limpieza básica."""
    df = pd.read_csv(
        data_path,
        na_values=["NA", "NaN", ""],
    )

    # Combinar columnas de fecha/hora en datetime
    if not {"year", "month", "day", "hour"}.issubset(df.columns):
        raise ValueError("El dataset debe contener las columnas year, month, day, hour")

    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].rename(columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}),
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    # Filtrar por estación
    if "station" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'station'")
    df = df[df["station"] == station].copy()
    if df.empty:
        raise ValueError(f"No hay datos para la estación '{station}'.")

    # Forward-fill de PM2.5 y limpieza básica de otras columnas
    df["PM2.5"] = df["PM2.5"].ffill()
    # Rellenos razonables
    for col in ["TEMP", "PRES", "WSPM", "RAIN"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Eliminar filas con PM2.5 sin datos
    df = df.dropna(subset=["PM2.5"])  # Asegurar que PM2.5 esté presente

    return df


def preprocess_features(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, Dict[str, Any]]:
    """Selecciona columnas clave, normaliza con MinMaxScaler, codifica 'wd' con one-hot y divide en train/test (por tiempo)."""
    required_cols = ["PM2.5", "TEMP", "PRES", "WSPM", "RAIN", "wd"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    # One-Hot para dirección del viento en todo el conjunto para asegurar columnas consistentes
    wd_dummies = pd.get_dummies(df["wd"].astype(str), prefix="wd")

    # Dividir por tiempo antes de ajustar escaladores (evitar leakage)
    n = len(df)
    train_end = int(n * train_ratio)
    if train_end <= 24:
        raise ValueError("Muy pocos datos para entrenamiento: se necesitan al menos 24*2 filas.")

    # Escaladores: uno para y (PM2.5) y otro para X numéricas
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()

    # Ajustar en entrenamiento y transformar en todo
    y_train_fit = df["PM2.5"].iloc[:train_end].values.reshape(-1, 1)
    scaler_y.fit(y_train_fit)
    y_scaled = scaler_y.transform(df["PM2.5"].values.reshape(-1, 1))

    X_numeric_cols = ["TEMP", "PRES", "WSPM", "RAIN"]
    X_train_fit = df[X_numeric_cols].iloc[:train_end].values
    scaler_X.fit(X_train_fit)
    X_scaled = scaler_X.transform(df[X_numeric_cols].values)

    # Matriz de features: incluir y_scaled como parte del contexto + numéricas escaladas + dummies
    features = np.hstack([y_scaled, X_scaled, wd_dummies.values])

    meta = {
        "train_end": train_end,
        "feature_names": ["PM2.5_scaled"] + X_numeric_cols + list(wd_dummies.columns),
        "scaler_X": scaler_X,
        "wd_columns": list(wd_dummies.columns),
    }

    return features, y_scaled.reshape(-1), scaler_y, meta


def build_sequences(features: np.ndarray, y_scaled: np.ndarray, seq_len: int = 24, train_end: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construye secuencias para LSTM con ventana deslizante de seq_len y divide en train/test por tiempo."""
    X, y = [], []
    n = len(features)
    for i in range(seq_len, n):
        X.append(features[i - seq_len:i, :])
        y.append(y_scaled[i])  # predecir el siguiente paso (la hora siguiente)
    X = np.array(X)
    y = np.array(y)

    # Dividir según train_end, ajustado por seq_len
    if train_end is None:
        train_end = int(n * 0.8)
    train_size_seq = max(1, train_end - seq_len)
    X_train, y_train = X[:train_size_seq], y[:train_size_seq]
    X_test, y_test = X[train_size_seq:], y[train_size_seq:]
    return X_train, y_train, X_test, y_test


def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """Construye un modelo LSTM para series temporales."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def plot_training_history(history, out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, len(history.history["loss"]) + 1), y=history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        sns.lineplot(x=range(1, len(history.history["val_loss"]) + 1), y=history.history["val_loss"], label="Val Loss")
    plt.xlabel("Épocas")
    plt.ylabel("MSE (pérdida)")
    plt.title("Entrenamiento LSTM - Curva de Pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_predictions(y_true_inv: np.ndarray, y_pred_inv: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv, label="Real PM2.5", alpha=0.8)
    plt.plot(y_pred_inv, label="Predicho PM2.5", alpha=0.8)
    plt.xlabel("Tiempo (horas)")
    plt.ylabel("PM2.5")
    plt.title("Comparación de Valores Reales vs Predichos (PM2.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_predictions(y_true_inv: np.ndarray, y_pred_inv: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true_inv, y_pred_inv)))
    mae = float(mean_absolute_error(y_true_inv, y_pred_inv))
    r2 = float(r2_score(y_true_inv, y_pred_inv))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def demo_predict_next_hour(model: Sequential, last_24_features: np.ndarray, scaler_y: MinMaxScaler) -> float:
    """Ejemplo de consulta: Dada una secuencia de 24 horas (features ya escaladas), predecir la próxima hora y devolver PM2.5 en escala original."""
    x_input = last_24_features[np.newaxis, :, :]  # (1, 24, n_features)
    y_pred_scaled = model.predict(x_input, verbose=0).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
    return float(y_pred)


def next_consulta_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("consulta_")]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except Exception:
            pass
    next_num = max(nums) + 1 if nums else 1
    consulta_dir = os.path.join(base_dir, f"consulta_{next_num:03d}")
    os.makedirs(consulta_dir, exist_ok=True)
    return consulta_dir


def safe_float(prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
    while True:
        raw = input(f"{prompt} "+(f"[default {default}]" if default is not None else "")+": ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"Valor demasiado bajo, mínimo {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Valor demasiado alto, máximo {max_val}")
                continue
            return val
        except ValueError:
            print("Entrada no válida, por favor ingrese un número.")


def choose_from_list(prompt: str, options: list, default: str = None) -> str:
    options_str = ", ".join(options)
    while True:
        raw = input(f"{prompt} ({options_str})" + (f" [default {default}]" if default else "") + ": ").strip()
        if raw == "" and default:
            return default
        if raw in options:
            return raw
        print("Opción no válida, intente nuevamente.")


def build_feature_row_from_user(meta: Dict[str, Any], scaler_X: MinMaxScaler, scaler_y: MinMaxScaler, allowed_wd: list) -> np.ndarray:
    print("\nIngrese datos de la hora actual:")
    pm25 = safe_float("PM2.5 (µg/m³)", default=35.0, min_val=0.0, max_val=1000.0)
    temp = safe_float("Temperatura (°C)", default=15.0, min_val=-40.0, max_val=45.0)
    pres = safe_float("Presión (hPa)", default=1013.0, min_val=900.0, max_val=1100.0)
    wspm = safe_float("Velocidad del viento (m/s)", default=2.0, min_val=0.0, max_val=40.0)
    rain = safe_float("Lluvia (mm/h)", default=0.0, min_val=0.0, max_val=100.0)
    wd = choose_from_list("Dirección del viento", allowed_wd, default=allowed_wd[0])

    # Escalar
    pm25_scaled = scaler_y.transform(np.array([[pm25]]))[0, 0]
    X_scaled = scaler_X.transform(np.array([[temp, pres, wspm, rain]]))[0]

    # One-hot de wd con columnas consistentes
    wd_vec = np.zeros(len(meta["wd_columns"]))
    colname = f"wd_{wd}"
    if colname in meta["wd_columns"]:
        wd_vec[meta["wd_columns"].index(colname)] = 1.0

    feature_row = np.hstack([[pm25_scaled], X_scaled, wd_vec])
    return feature_row


def interactive_loop(df: pd.DataFrame, features: np.ndarray, model: Sequential, scaler_y: MinMaxScaler, meta: Dict[str, Any], graficos_dir: str) -> None:
    allowed_wd = [c.replace("wd_", "") for c in meta["wd_columns"]]
    scaler_X: MinMaxScaler = meta["scaler_X"]
    while True:
        print("\n================ CONSULTAS INTERACTIVAS ================")
        print("1) Predicción inmediata con últimas 24 horas del dataset")
        print("2) Predicción ingresando valores de la hora actual")
        print("3) Salir")
        choice = input("Seleccione una opción (1-3): ").strip()

        if choice == "1":
            consulta_dir = next_consulta_dir(graficos_dir)
            last_24 = features[-24:, :]
            pred_val = demo_predict_next_hour(model, last_24, scaler_y)
            print(f"Predicción PM2.5 próxima hora: {pred_val:.2f} µg/m³")
            # Graficar últimas 24 y punto predicho
            plt.figure(figsize=(10, 4))
            pm24_inv = scaler_y.inverse_transform(features[-24:, 0].reshape(-1, 1)).reshape(-1)
            plt.plot(range(24), pm24_inv, label="Últimas 24h PM2.5", marker="o")
            plt.scatter([24], [pred_val], color="red", label="Predicción próxima hora")
            plt.xlabel("Hora desplazada")
            plt.ylabel("PM2.5 (µg/m³)")
            plt.title("Consulta: Últimas 24h + Predicción próxima hora")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(consulta_dir, "consulta_pred_next_hour.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Gráfico guardado en: {out_path}")
        elif choice == "2":
            consulta_dir = next_consulta_dir(graficos_dir)
            feature_row = build_feature_row_from_user(meta, scaler_X, scaler_y, allowed_wd)
            # Construir secuencia: últimas 23 horas + hora actual del usuario
            seq = np.vstack([features[-23:, :], feature_row])
            pred_val = demo_predict_next_hour(model, seq, scaler_y)
            print(f"Predicción PM2.5 próxima hora (con entrada del usuario): {pred_val:.2f} µg/m³")
            # Graficar
            plt.figure(figsize=(10, 4))
            pm23_inv = scaler_y.inverse_transform(features[-23:, 0].reshape(-1, 1)).reshape(-1)
            current_pm = scaler_y.inverse_transform(np.array([[feature_row[0]]])).reshape(-1)[0]
            series = list(pm23_inv) + [current_pm]
            plt.plot(range(24), series, label="Secuencia usada (23h dataset + 1h usuario)", marker="o")
            plt.scatter([24], [pred_val], color="red", label="Predicción próxima hora")
            plt.xlabel("Hora desplazada")
            plt.ylabel("PM2.5 (µg/m³)")
            plt.title("Consulta usuario: Secuencia + Predicción")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(consulta_dir, "consulta_usuario_pred.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Gráfico guardado en: {out_path}")
        elif choice == "3":
            print("Saliendo del modo interactivo...")
            break
        else:
            print("Opción no válida. Intente nuevamente.")


def main():
    set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dirs = ensure_output_dirs(base_dir)
    data_path = os.path.join(base_dir, "data", "Beijing Multisite air Quality data.csv")

    print("==============================")
    print("🌫️  LSTM Predicción de Calidad del Aire (PM2.5)")
    print("==============================")
    print(f"Data: {data_path}")
    print("Estación: Aotizhongxin")

    t0 = time.time()
    df = load_and_clean_data(data_path, station="Aotizhongxin")
    print(f"Filas tras limpieza: {len(df)}")

    features, y_scaled, scaler_y, meta = preprocess_features(df, train_ratio=0.8)
    X_train, y_train, X_test, y_test = build_sequences(features, y_scaled, seq_len=24, train_end=meta["train_end"])

    print(f"Secuencias de entrenamiento: {X_train.shape}, prueba: {X_test.shape}")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary(print_fn=lambda x: print("[Model] " + x))

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    t1 = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=1,
    )
    t2 = time.time()

    # Gráfico de entrenamiento
    loss_path = os.path.join(out_dirs["graficos"], "lstm_airquality_loss.png")
    plot_training_history(history, loss_path)
    print(f"Gráfico de pérdida guardado en: {loss_path}")

    # Predicciones sobre test
    y_pred_test_scaled = model.predict(X_test, verbose=0)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).reshape(-1)
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # Evaluación
    metrics = evaluate_predictions(y_test_inv, y_pred_test)
    print("\nMétricas en conjunto de prueba:")
    print(json.dumps(metrics, indent=2))

    # Gráfico de resultados
    pred_path = os.path.join(out_dirs["graficos"], "lstm_airquality_pred_vs_real.png")
    plot_predictions(y_test_inv, y_pred_test, pred_path)
    print(f"Gráfico real vs predicho guardado en: {pred_path}")

    # Consulta inmediata: usar las últimas 24 horas del dataset para estimar la próxima
    last_24 = features[-24:, :]  # ya escaladas y con dummies
    next_pm25 = demo_predict_next_hour(model, last_24, scaler_y)
    print(f"\nConsulta inmediata: Predicción PM2.5 para la próxima hora = {next_pm25:.2f}")

    print(f"\nTiempo de carga y preparación: {t1 - t0:.2f}s")
    print(f"Tiempo de entrenamiento: {t2 - t1:.2f}s")
    print(f"Tiempo total: {time.time() - t0:.2f}s")

    # Modo interactivo
    interactive_loop(df, features, model, scaler_y, meta, out_dirs["graficos"])


if __name__ == "__main__":
    main()