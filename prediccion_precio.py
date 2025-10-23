"""
Script to build and evaluate a neural network for predicting house prices.

This module illustrates how to train a feed‑forward neural network on
a tabular regression task.  By default it works with the Boston
housing data set included in this project (``data/Boston.csv``),
which contains the median value of owner‑occupied homes (target) and
13 predictor variables such as per‑capita crime rate, average number
of rooms and property tax rate.  Nevertheless, the code is written
to accept any CSV file with continuous or categorical predictors and
a single numeric target (assumed to be the last column unless a
column named ``precio`` is present).

The work flow followed by this script is:

1. Load the data from a local CSV file.
2. Optionally rename the target column to ``precio`` (if it already
   exists the script will use it directly).
3. One‑hot encode any categorical variables and standardise all
   numerical predictors.
4. Split the data into train and test subsets.
5. Define and train a multi‑layer perceptron for regression using a
   simple implementation provided in ``neural_network.py``.
6. Evaluate the model on both train and test sets (RMSE and R²).
7. Plot the training loss curve and a scatter plot comparing
   predicted and actual prices on the test set.
8. Display a summary table of evaluation metrics.

Running this module as a script will execute the full pipeline and
store plots in the current working directory.  It can also be
imported and extended for more sophisticated experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from neural_network import NeuralNetwork


def load_housing_data(path: str) -> pd.DataFrame:
    """Load a housing data set from a local CSV file.

    This function is flexible and can ingest any comma‑separated data
    set.  If the file contains a header row it will be preserved;
    otherwise the first row will be interpreted as data.  When the
    Boston housing data set is used (as included in this project under
    ``data/Boston.csv``) the columns correspond to the original
    attributes from the UCI repository: ``CRIM``, ``ZN``, ``INDUS``,
    ``CHAS``, ``NOX``, ``RM``, ``AGE``, ``DIS``, ``RAD``, ``TAX``,
    ``PTRATIO``, ``LSTAT`` and the target ``MEDV`` (median value of
    owner‑occupied homes in $1000s).

    Parameters
    ----------
    path : str
        Path to the local CSV file.  Relative paths are resolved
        relative to the current working directory.  If the file is
        distributed with this project it resides under
        ``redes‑neuronales/data``.

    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the loaded data.  Column names are
        inferred from the file header if present.
    """
    # Read CSV; header is automatically detected.  The local file may
    # reside inside the package folder.  ``on_bad_lines='skip'`` is
    # used to gracefully skip malformed rows if any.
    df = pd.read_csv(path, sep=",", header='infer', on_bad_lines='skip')
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Preprocess a generic housing data set for regression.

    The function performs the following steps:

    * Splits the data into predictors ``X_df`` and target ``y``.  The
      target is assumed to be the last column in ``df`` (e.g. ``MEDV``
      for the Boston housing data or ``precio`` in the Saratoga
      example).  Users can override this behaviour by renaming the
      target column to ``precio`` before calling this function.
    * Applies one‑hot encoding to any categorical (object) variables.
    * Standardises all numeric features to zero mean and unit variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data with predictors and a single target column.

    Returns
    -------
    X : np.ndarray
        Normalised and (if necessary) encoded feature matrix.
    y : np.ndarray
        Target vector reshaped to (n_samples, 1).
    stats : dict
        Dictionary containing the mean and standard deviation for
        scaling (keys: ``'mean'``, ``'std'``) and the list of feature
        names after encoding (key ``'columns'``).
    """
    # Identify target as the last column if ``precio`` not present.
    if 'precio' in df.columns:
        target_col = 'precio'
    else:
        target_col = df.columns[-1]
    y = df[[target_col]].values.astype(float)
    X_df = df.drop(columns=[target_col])
    # Identify categorical and numerical columns
    cat_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    # One‑hot encode categoricals (if any)
    X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
    # Standardise numerical columns
    mean = X_encoded.mean(axis=0)
    std = X_encoded.std(axis=0).replace(0, 1)
    X_norm = (X_encoded - mean) / std
    return (
        X_norm.values.astype(float),
        y.astype(float),
        {
            'mean': mean.values,
            'std': std.values,
            'columns': X_encoded.columns.tolist(),
        },
    )


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float, optional
        Fraction of the data to assign to the test set.  Default is 0.2.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Arrays corresponding to the split data.
    """
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    n_test = int(len(indices) * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main() -> None:
    """Load data, train a regression model, evaluate and plot results.

    This entry point reads the housing data from a local CSV file
    bundled with the project (``data/Boston.csv``).  It then
    normalises the predictors, splits the data into training and test
    partitions, trains a multi‑layer perceptron for regression and
    reports performance on both splits.  Diagnostic plots of the
    training loss and the predicted vs. actual prices are saved to
    disk.
    """
    import os
    # Construct path to the Boston housing data in the repository
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "Boston.csv")
    df = load_housing_data(data_path)
    # Preprocess the data: normalise numeric features and one‑hot encode
    # categoricals.  The function automatically selects the last column
    # as the target when the column ``precio`` is not present.
    X, y, stats = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_features = X_train.shape[1]
    # Define neural network architecture
    layer_sizes = [n_features, 32, 16, 1]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_activation="relu",
        output_activation="linear",
        loss="mse",
        learning_rate=0.001,
        random_state=42,
    )
    history = nn.fit(X_train, y_train, epochs=800, verbose=False)
    # Evaluate
    train_metrics = nn.evaluate_regression(X_train, y_train)
    test_metrics = nn.evaluate_regression(X_test, y_test)
    # Create loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Loss de entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (MSE)")
    plt.title("Curva de pérdida durante el entrenamiento")
    plt.legend()
    loss_plot_path = "housing_loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    # Scatter plot of predictions vs actual on test set
    y_pred_test = nn.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    max_val = max(y_test.max(), y_pred_test.max())
    min_val = min(y_test.min(), y_pred_test.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("Predicciones vs. valores reales (conjunto de test)")
    scatter_plot_path = "housing_pred_vs_actual.png"
    plt.tight_layout()
    plt.savefig(scatter_plot_path)
    plt.close()
    # Display metrics in a table
    results_df = pd.DataFrame([
        ["RMSE", train_metrics["rmse"], test_metrics["rmse"]],
        ["R2", train_metrics["r2"], test_metrics["r2"]],
    ], columns=["Métrica", "Entrenamiento", "Test"])
    print("\nResumen de métricas de regresión:\n")
    print(results_df.to_string(index=False, formatters={
        "Entrenamiento": lambda x: f"{x:.3f}",
        "Test": lambda x: f"{x:.3f}",
    }))
    print("\nGráficas guardadas en archivos:")
    print(f" - Curva de pérdida: {loss_plot_path}")
    print(f" - Predicción vs. real: {scatter_plot_path}")


if __name__ == "__main__":
    main()
