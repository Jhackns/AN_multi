"""
Script to train a neural network for diabetes diagnosis.

This module addresses a binary classification problem using the
Pima Indians diabetes data set.  Each record consists of several
medical measurements (such as glucose concentration, blood pressure,
body mass index) and the target indicates whether the patient has
diabetes (1) or not (0).  The data set is included locally in this
project under ``data/diabetes.csv`` and does not contain a header
row; appropriate column names are therefore assigned by the
``load_diabetes_data`` function.

The neural network implemented here does not rely on external machine
learning frameworks and therefore works in restricted environments.

Work flow:

1. Load the data from the local CSV file and assign column names.
2. Replace invalid zero values in selected columns with the column
   mean.  Some measurements (glucose, blood pressure, skin thickness,
   insulin and BMI) cannot legitimately be zero.
3. Standardise the features.
4. Split the data into training and test sets.
5. Train a multi‑layer perceptron with sigmoid output and
   binary cross‑entropy loss.
6. Evaluate the model on training and test data (accuracy, precision,
   recall and F1).
7. Plot the loss curve and the confusion matrix on the test set.
8. Display a summary table of classification metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from neural_network import NeuralNetwork


def load_diabetes_data(path: str, names: list[str] | None = None) -> pd.DataFrame:
    """Load a diabetes data set from a local CSV file.

    By default this function attempts to infer whether the file
    contains a header row.  If a list of column names is supplied via
    ``names`` it will be used instead.  The Pima Indians diabetes
    data bundled with this project (``data/diabetes.csv``) does not
    contain a header row, so appropriate column names must be
    provided.

    Parameters
    ----------
    path : str
        Path to the local CSV file.
    names : list of str, optional
        Column names to assign when the file lacks a header.  If
        ``None`` (default) pandas will attempt to read an existing
        header row.

    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data.
    """
    df = pd.read_csv(path, sep=",", header=None if names is not None else 'infer', names=names)
    return df


def replace_zero_with_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace zero values in specified columns with the column mean.

    In the Pima Indians data set certain variables cannot be zero (for
    example blood pressure).  This helper replaces zeros with the
    arithmetic mean of the column, computed ignoring zeros.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame to process.
    columns : list of str
        Column names in which zeros should be replaced.

    Returns
    -------
    df : pandas.DataFrame
        The data frame with zero values replaced.
    """
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
    """Preprocess the diabetes data.

    Separates features and labels, replaces invalid zeros, scales the
    features and returns the resulting matrices.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw diabetes data.

    Returns
    -------
    X : np.ndarray
        Normalised feature matrix.
    y : np.ndarray
        Target vector (0/1) reshaped to (n_samples, 1).
    stats : dict
        Dictionary containing the mean and standard deviation for
        scaling (keys: ``'mean'``, ``'std'``).
    """
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
    """Split arrays into random train and test subsets (same as in housing script)."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    n_test = int(len(indices) * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix for binary classification.

    Returns
    -------
    cm : np.ndarray, shape (2, 2)
        Confusion matrix with rows corresponding to true classes and
        columns to predicted classes::

            cm[0, 0] = TN, cm[0, 1] = FP
            cm[1, 0] = FN, cm[1, 1] = TP
    """
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def main() -> None:
    """Load data, train a classification model, evaluate and plot results.

    This entry point reads the Pima Indians diabetes data from a
    local CSV file included in this project (``data/diabetes.csv``),
    assigns meaningful column names, normalises the features, splits
    the data into training and test partitions, trains a multi‑layer
    perceptron for binary classification and reports performance on
    both splits.  Diagnostic plots of the training loss and the
    confusion matrix are saved to disk.
    """
    import os
    # Build the path to the local diabetes data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "diabetes.csv")
    # Column names for the Pima Indians data set (no header in file)
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
    df = load_diabetes_data(data_path, names=columns)
    X, y, stats = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_features = X_train.shape[1]
    # Define neural network architecture for classification
    layer_sizes = [n_features, 16, 8, 1]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_activation="relu",
        output_activation="sigmoid",
        loss="binary_crossentropy",
        learning_rate=0.01,
        random_state=42,
    )
    # Train for a moderate number of epochs to ensure convergence without
    # excessive computation time.  If you wish to improve performance
    # further you can increase this value.
    history = nn.fit(X_train, y_train, epochs=500, verbose=False)
    # Evaluate metrics
    train_metrics = nn.evaluate_classification(X_train, y_train)
    test_metrics = nn.evaluate_classification(X_test, y_test)
    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Loss de entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (binary crossentropy)")
    plt.title("Curva de pérdida durante el entrenamiento - Diabetes")
    plt.legend()
    loss_plot_path = "diabetes_loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    # Confusion matrix on test set
    y_pred_probs = nn.predict(X_test)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusión - Conjunto de test")
    cm_plot_path = "diabetes_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()
    # Display metrics table
    results_df = pd.DataFrame([
        ["Accuracy", train_metrics["accuracy"], test_metrics["accuracy"]],
        ["Precisión", train_metrics["precision"], test_metrics["precision"]],
        ["Recall", train_metrics["recall"], test_metrics["recall"]],
        ["F1", train_metrics["f1"], test_metrics["f1"]],
    ], columns=["Métrica", "Entrenamiento", "Test"])
    print("\nResumen de métricas de clasificación:\n")
    print(results_df.to_string(index=False, formatters={
        "Entrenamiento": lambda x: f"{x:.3f}",
        "Test": lambda x: f"{x:.3f}",
    }))
    print("\nGráficas guardadas en archivos:")
    print(f" - Curva de pérdida: {loss_plot_path}")
    print(f" - Matriz de confusión: {cm_plot_path}")


if __name__ == "__main__":
    main()
