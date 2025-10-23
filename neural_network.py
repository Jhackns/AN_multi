"""
Generic feed‑forward neural network implementation using only NumPy.

This module exposes a ``NeuralNetwork`` class that can be used for both
regression and binary classification tasks.  It supports an arbitrary
number of hidden layers with ReLU or sigmoid activations and can train
using batch gradient descent.  The network is deliberately simple: it
does not rely on external machine learning libraries (such as
scikit‑learn or TensorFlow) and therefore runs in environments where
those packages are unavailable.  Despite its simplicity, the
implementation demonstrates the core concepts of backpropagation and
gradient descent.

Example
-------

Create a network for regression with one hidden layer::

    import numpy as np
    from neural_network import NeuralNetwork

    # Generate synthetic data: y = x1 + 2 * x2 + noise
    rng = np.random.RandomState(42)
    X = rng.rand(200, 2)
    y = X[:, 0:1] + 2 * X[:, 1:2] + 0.1 * rng.randn(200, 1)

    # Define network with one hidden layer of 10 units
    nn = NeuralNetwork(layer_sizes=[2, 10, 1],
                       hidden_activation="relu",
                       output_activation="linear",
                       loss="mse",
                       learning_rate=0.01,
                       random_state=42)

    history = nn.fit(X, y, epochs=500, verbose=False)

    # Predict on new data
    y_pred = nn.predict(X)
    print("RMSE:", np.sqrt(np.mean((y - y_pred)**2)))

For binary classification use a sigmoid output and cross‑entropy loss::

    import numpy as np
    from neural_network import NeuralNetwork

    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(layer_sizes=[2, 4, 1],
                       hidden_activation="relu",
                       output_activation="sigmoid",
                       loss="binary_crossentropy",
                       learning_rate=0.1,
                       random_state=0)

    history = nn.fit(X, y, epochs=2000, verbose=False)
    print("Predictions:", nn.predict(X))

Notes
-----
* Only two activation functions are supported on hidden layers: ReLU
  and sigmoid.  The output layer can use either linear (for
  regression) or sigmoid (for binary classification).  Additional
  activations can be added by extending the `_activation` and
  `_activation_derivative` methods.
* Loss functions supported are mean squared error (``mse``) for
  regression and binary cross entropy (``binary_crossentropy``) for
  classification.  The derivative of the loss with respect to the
  network output is computed internally.
* Mini‑batch updates are not currently implemented; the network uses
  full batch gradient descent.  For most moderate sized data sets
  (hundreds or thousands of samples) this is sufficient.  Larger
  problems may require mini‑batch support.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict


class NeuralNetwork:
    """A simple fully connected neural network.

    Parameters
    ----------
    layer_sizes : List[int]
        List containing the number of units in each layer.  The first
        element corresponds to the number of input features and the last
        element to the number of outputs.  Intermediate values define
        hidden layers.
    hidden_activation : str, optional
        Activation function used in hidden layers.  Supported values
        include ``"relu"`` and ``"sigmoid"``.  Default is ``"relu"``.
    output_activation : str, optional
        Activation function used in the output layer.  Supported values
        include ``"linear"`` and ``"sigmoid"``.  For regression tasks
        use ``"linear"``; for binary classification use ``"sigmoid"``.
    loss : str, optional
        Loss function to optimize.  Supported values include
        ``"mse"`` (mean squared error) and ``"binary_crossentropy"``.
        Default is ``"mse"``.
    learning_rate : float, optional
        Step size used during gradient descent updates.  Default is
        ``0.01``.
    random_state : int or None, optional
        Seed for random number generation.  If provided the network's
        weights are initialised deterministically.

    Attributes
    ----------
    weights : List[np.ndarray]
        Weight matrices for each layer.  Each element has shape
        `(n_features_prev, n_features_current)`.
    biases : List[np.ndarray]
        Bias vectors for each layer.  Each element has shape
        `(1, n_features_current)`.
    history_ : Dict[str, List[float]]
        Training history storing the loss value per epoch.  It is
        populated after calling :meth:`fit`.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        hidden_activation: str = "relu",
        output_activation: str = "linear",
        loss: str = "mse",
        learning_rate: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output layers")
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.learning_rate = learning_rate
        self.random_state = random_state
        # Initialise random generator
        self._rng = np.random.RandomState(random_state)
        # Initialise weights and biases
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier/Glorot initialisation for hidden layers
            limit = np.sqrt(6 / (in_size + out_size))
            W = self._rng.uniform(-limit, limit, size=(in_size, out_size))
            b = np.zeros((1, out_size))
            self.weights.append(W)
            self.biases.append(b)
        self.history_: Dict[str, List[float]] = {"loss": []}

    def _activation(self, z: np.ndarray, kind: str) -> np.ndarray:
        """Apply activation function element‑wise."""
        if kind == "relu":
            return np.maximum(0, z)
        elif kind == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif kind == "linear":
            return z
        else:
            raise ValueError(f"Unsupported activation '{kind}'")

    def _activation_derivative(self, z: np.ndarray, kind: str) -> np.ndarray:
        """Derivative of activation function element‑wise."""
        if kind == "relu":
            return (z > 0).astype(float)
        elif kind == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif kind == "linear":
            return np.ones_like(z)
        else:
            raise ValueError(f"Unsupported activation '{kind}'")

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the loss for the current predictions."""
        m = y_true.shape[0]
        if self.loss == "mse":
            return 0.5 * np.mean((y_true - y_pred) ** 2)
        elif self.loss == "binary_crossentropy":
            # Avoid log(0) by clipping
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        else:
            raise ValueError(f"Unsupported loss '{self.loss}'")

    def _loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of the loss function with respect to the output layer activations."""
        m = y_true.shape[0]
        if self.loss == "mse":
            return (y_pred - y_true) / m
        elif self.loss == "binary_crossentropy":
            # For cross entropy with sigmoid output, derivative simplifies to (y_pred - y_true)/m
            return (y_pred - y_true) / m
        else:
            raise ValueError(f"Unsupported loss '{self.loss}'")

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute forward pass and return list of z and a for each layer.

        Returns
        -------
        zs : list of np.ndarray
            Pre‑activation values for each layer (excluding input layer).
        activations : list of np.ndarray
            Activations for each layer including input.  The first element is
            X itself; the last element is the network output.
        """
        activations = [X]
        zs = []
        a = X
        # Hidden layers
        for idx, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = np.dot(a, W) + b
            a = self._activation(z, self.hidden_activation)
            zs.append(z)
            activations.append(a)
        # Output layer
        W_out, b_out = self.weights[-1], self.biases[-1]
        z_out = np.dot(a, W_out) + b_out
        y_pred = self._activation(z_out, self.output_activation)
        zs.append(z_out)
        activations.append(y_pred)
        return zs, activations

    def backward(self, zs: List[np.ndarray], activations: List[np.ndarray], y_true: np.ndarray) -> None:
        """Perform backpropagation and update weights and biases."""
        # Number of layers (excluding input)
        L = len(self.weights)
        # Initialize list for deltas
        deltas: List[np.ndarray] = [None] * L
        # Compute derivative of loss w.r.t. output activation
        y_pred = activations[-1]
        delta = self._loss_derivative(y_true, y_pred) * self._activation_derivative(zs[-1], self.output_activation)
        deltas[-1] = delta
        # Backpropagate through hidden layers
        for l in reversed(range(L - 1)):
            z = zs[l]
            W_next = self.weights[l + 1]
            delta_next = deltas[l + 1]
            delta = np.dot(delta_next, W_next.T) * self._activation_derivative(z, self.hidden_activation)
            deltas[l] = delta
        # Update weights and biases
        for l in range(L):
            a_prev = activations[l]
            delta_l = deltas[l]
            dW = np.dot(a_prev.T, delta_l)
            db = np.sum(delta_l, axis=0, keepdims=True)
            self.weights[l] -= self.learning_rate * dW
            self.biases[l] -= self.learning_rate * db

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = False) -> Dict[str, List[float]]:
        """Train the neural network.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples, n_outputs)
            Target values.
        epochs : int, optional
            Number of training iterations.  Default is 1000.
        verbose : bool, optional
            If ``True``, print the loss every 100 epochs.

        Returns
        -------
        history : dict
            Dictionary containing the loss per epoch.
        """
        self.history_ = {"loss": []}
        for epoch in range(1, epochs + 1):
            zs, activations = self.forward(X)
            loss = self._compute_loss(y, activations[-1])
            self.history_["loss"].append(loss)
            self.backward(zs, activations, y)
            if verbose and (epoch % 100 == 0 or epoch == 1):
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
        return self.history_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output for given input X.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        y_pred : np.ndarray
            Predicted outputs.
        """
        _, activations = self.forward(X)
        return activations[-1]

    def evaluate_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate regression performance metrics.

        Computes the root mean squared error (RMSE) and the coefficient of
        determination R^2 on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True targets.

        Returns
        -------
        metrics : dict
            Dictionary with keys ``'rmse'`` and ``'r2'``.
        """
        y_pred = self.predict(X)
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return {"rmse": rmse, "r2": r2}

    def evaluate_classification(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate binary classification performance metrics.

        Computes accuracy, precision, recall and F1 score on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True binary targets (0 or 1).
        threshold : float, optional
            Decision threshold for converting probabilities into class labels.

        Returns
        -------
        metrics : dict
            Dictionary with keys ``'accuracy'``, ``'precision'``,
            ``'recall'`` and ``'f1'``.
        """
        probs = self.predict(X)
        y_pred = (probs >= threshold).astype(int)
        y_true = y.astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
