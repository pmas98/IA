import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, hidden_layers=(2, ), learning_rate=1e-2, epochs=1000,
                 activation='sigmoid', task='regression'):

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.task = task
        self.weights = []
        self.biases = []
        self.cost_history = []
        self.val_cost_history = []
        self.fitted = False 

    def _initialize_weights(self, n_features, n_outputs):
        np.random.seed(42)
        layer_sizes = [n_features] + list(self.hidden_layers) + [n_outputs]
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            w = np.random.uniform(-0.5, 0.5, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _activation(self, z, derivative=False):
        if self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s) if derivative else s
        if self.activation == 'tanh':
            t = np.tanh(z)
            return (1 - t**2) if derivative else t

    def _forward(self, X):
        a = X 
        activations, zs = [a], []
        for idx in range(len(self.weights)):
            z = a.dot(self.weights[idx]) + self.biases[idx]
            zs.append(z)
            if idx < len(self.weights) - 1:
                a = self._activation(z)
            else:
                a = z if self.task == 'regression' else 1 / (1 + np.exp(-z))
            activations.append(a)
        return activations, zs

    def _compute_cost(self, y_true, y_pred):
        if hasattr(self, 'y_std') and hasattr(self, 'y_mean'):
            y_true_orig = y_true * self.y_std + self.y_mean
            y_pred_orig = y_pred * self.y_std + self.y_mean
            return np.mean((y_true_orig - y_pred_orig)**2)
        else:
            return np.mean((y_true - y_pred)**2)

    def _backward(self, X, y, activations, zs):
        m = X.shape[0]
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y

        dw[-1] = activations[-2].T.dot(delta) / m 
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m

        for l in range(len(self.weights)-2, -1, -1):
            delta = delta.dot(self.weights[l+1].T) * self._activation(zs[l], derivative=True)
            dw[l] = activations[l].T.dot(delta) / m
            db[l] = np.sum(delta, axis=0, keepdims=True) / m
        return dw, db

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, patience=20, y_mean=None, y_std=None):
        y = y.reshape(-1,1)
        if X_val is not None: y_val = y_val.reshape(-1,1)
        n_samples, n_features = X.shape
        n_outputs = y.shape[1] 

        if y_mean is not None and y_std is not None:
            self.y_mean = y_mean
            self.y_std = y_std

        self._initialize_weights(n_features, n_outputs)

        best_val = np.inf; counter = 0; best_w = None; best_b = None

        for _ in range(1, self.epochs+1):
            acts, zs = self._forward(X)
            cost = self._compute_cost(y, acts[-1]); self.cost_history.append(cost)

            if X_val is not None:
                val_acts, _ = self._forward(X_val)
                val_cost = self._compute_cost(y_val, val_acts[-1]); self.val_cost_history.append(val_cost)

                if early_stopping:
                    if val_cost < best_val:
                        best_val = val_cost
                        best_w = [w.copy() for w in self.weights] 
                        best_b = [b.copy() for b in self.biases] 
                        counter = 0

                    else:
                        counter += 1
                    if counter >= patience:
                        self.weights, self.biases = best_w, best_b
                        break

            dw, db = self._backward(X, y, acts, zs)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dw[i]
                self.biases[i] -= self.learning_rate * db[i]

        self.fitted = True
        return self

    def predict(self, X):
        return self._forward(X)[0][-1].flatten()

    def plot_learning_curve(self, title='Curva de Aprendizado'):
        plt.figure(figsize=(8,5))
        epochs = np.arange(1, len(self.cost_history)+1)

        plt.plot(epochs, self.cost_history, label='Treino')
        if self.val_cost_history: plt.plot(epochs, self.val_cost_history, label='Validação')
        plt.title(title)
        plt.xlabel('Épocas')
        plt.ylabel('MSE' if self.task == 'regression' else 'Custo')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self, X, y):
        y_true = y.copy()
        if hasattr(y_true, 'ndim') and y_true.ndim > 1:
            y_true = y_true.flatten()

        y_pred = self.predict(X)
        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        errors = y_true - y_pred
        mse  = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae  = np.mean(np.abs(errors))

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) 
        r2    = 1 - ss_res / ss_tot if ss_tot > 0 else 0 

        return {
            'mean_absolute_error':      mae,
            'mean_squared_error':       mse,
            'root_mean_squared_error':  rmse,
            'r2_score':                 r2
        }
