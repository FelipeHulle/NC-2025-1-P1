import numpy as np
from numpy.linalg import pinv

class ELM:
    def __init__(self, n_hidden_neurons, activation='sigmoid'):
        self.n_hidden = n_hidden_neurons
        self.activation = activation

    def _activation(self, x):
        if self.activation == 'sigmoid':
            return 1.0 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function.")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.input_weights = np.random.normal(size=(self.n_hidden, n_features))
        self.biases = np.random.normal(size=(self.n_hidden, ))

        H = self._activation(np.dot(X, self.input_weights.T) + self.biases)

        self.classes_ = np.unique(y)
        T = np.zeros((n_samples, n_classes))
        for i, label in enumerate(self.classes_):
            T[y == label, i] = 1

        self.output_weights = np.dot(pinv(H), T)

    def predict(self, X):
        H = self._activation(np.dot(X, self.input_weights.T) + self.biases)
        output = np.dot(H, self.output_weights)
        return self.classes_[np.argmax(output, axis=1)]
