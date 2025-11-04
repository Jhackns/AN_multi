from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict

class NeuralNetwork:
    
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
       
        if kind == "relu":
            return np.maximum(0, z)
        elif kind == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif kind == "linear":
            return z
        else:
            raise ValueError(f"Unsupported activation '{kind}'")

    def _activation_derivative(self, z: np.ndarray, kind: str) -> np.ndarray:
        
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
        
        m = y_true.shape[0]
        if self.loss == "mse":
            return (y_pred - y_true) / m
        elif self.loss == "binary_crossentropy":
            # For cross entropy with sigmoid output, derivative simplifies to (y_pred - y_true)/m
            return (y_pred - y_true) / m
        else:
            raise ValueError(f"Unsupported loss '{self.loss}'")

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
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
       
        _, activations = self.forward(X)
        return activations[-1]

    def evaluate_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
       
        y_pred = self.predict(X)
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return {"rmse": rmse, "r2": r2}

    def evaluate_classification(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        
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
