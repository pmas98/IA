import numpy as np

class MLP:
    def __init__(self, hidden_layers=(100, 50), learning_rate=0.001, epochs=1000,
                 batch_size=32, random_seed=42, tol=1e-6, activation='linear',
                 # Define clipping bounds as parameters if needed, or keep fixed
                 error_clip_low=-1e10, error_clip_high=1e10):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.tol = tol
        self.activation = activation
        self.error_clip_low = error_clip_low
        self.error_clip_high = error_clip_high

        self.weights = []
        self.biases = []
        self.cost_ = []
        self.val_cost_ = []
        self.fitted = False
        self.confusion_matrix = None

        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def _initialize_weights(self, n_features, n_classes):
        np.random.seed(self.random_seed)
        layer_sizes = [n_features] + list(self.hidden_layers) + [n_classes]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            if self.activation == 'relu':
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            b = np.random.randn(layer_sizes[i+1]) * 0.01
            self.weights.append(w)
            self.biases.append(b)

    def _activation_function(self, z, derivative=False):
        if self.activation == 'sigmoid':
            z_clipped = np.clip(z, -500, 500)
            s = 1.0 / (1.0 + np.exp(-z_clipped))
            if derivative:
                return s * (1 - s)
            return s
        elif self.activation == 'linear':
            if derivative:
                return np.ones_like(z)
            return z
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _forward_pass(self, X):
        activations = [X]
        pre_activations = []
        a = X
        for i in range(len(self.weights)):
            z = a.dot(self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                 a = self._activation_function(z)
            else:
                 a = z
            activations.append(a)
        return activations, pre_activations

    # --- MODIFIED METHOD ---
    def _compute_cost(self, y_true, y_pred):
        """Computes the cost (MSE/2) using clipped errors, similar to Adaline."""
        errors = y_true - y_pred
        errors_clipped = np.clip(errors, self.error_clip_low, self.error_clip_high)
        cost = np.mean(errors_clipped**2) / 2.0
        return cost

    def _backward_pass(self, X, y, activations, pre_activations):
        m = X.shape[0]
        n_layers = len(self.weights)
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y 

        dw[n_layers-1] = activations[n_layers-1].T.dot(delta) / m
        db[n_layers-1] = np.sum(delta, axis=0) / m

        for l in reversed(range(n_layers - 1)):
            delta = delta.dot(self.weights[l+1].T) * self._activation_function(pre_activations[l], derivative=True)

            dw[l] = activations[l].T.dot(delta) / m
            db[l] = np.sum(delta, axis=0) / m

        return dw, db

    def _normalize_data(self, X, y=None, fit=False):
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std == 0] = 1.0
            if y is not None:
                self.y_mean = np.mean(y, axis=0)
                self.y_std = np.std(y, axis=0)
                self.y_std[self.y_std == 0] = 1.0

        if self.X_mean is None or self.X_std is None:
             raise Exception("Normalization parameters not set. Call fit first.")

        Xn = (X - self.X_mean) / self.X_std

        if y is not None:
            if self.y_mean is None or self.y_std is None:
                 raise Exception("Target normalization parameters not set. Call fit first.")
            yn = (y - self.y_mean) / self.y_std
            return Xn, yn
        return Xn

    def _denormalize_predictions(self, y_norm):
        if self.y_mean is None or self.y_std is None:
             raise Exception("Target normalization parameters not set. Call fit first.")
        return y_norm * self.y_std + self.y_mean

    def fit(self, X, y, X_val=None, y_val=None):
        y_proc = y.copy()
        if y_proc.ndim == 1:
            y_proc = y_proc.reshape(-1, 1)

        Xn, yn = self._normalize_data(X.copy(), y_proc, fit=True)

        Xvn, yvn = None, None
        if X_val is not None and y_val is not None:
             y_val_proc = y_val.copy()
             if y_val_proc.ndim == 1:
                  y_val_proc = y_val_proc.reshape(-1, 1)
             Xvn, yvn = self._normalize_data(X_val.copy(), y_val_proc, fit=False)


        n_samples, n_features = Xn.shape
        n_outputs = yn.shape[1]
        self._initialize_weights(n_features, n_outputs)

        self.cost_, self.val_cost_ = [], []
        lr = self.learning_rate 

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            Xs, ys = Xn[perm], yn[perm]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                Xb, yb = Xs[start:end], ys[start:end]

                activations, pre_activations = self._forward_pass(Xb)

                dw, db = self._backward_pass(Xb, yb, activations, pre_activations)

                for i in range(len(self.weights)):
                    self.weights[i] -= lr * dw[i]
                    self.biases[i]  -= lr * db[i]

            act_full, _ = self._forward_pass(Xn)
            cost = self._compute_cost(yn, act_full[-1])
            self.cost_.append(cost)

            if Xvn is not None and yvn is not None:
                act_val, _ = self._forward_pass(Xvn)
                vcost = self._compute_cost(yvn, act_val[-1])
                self.val_cost_.append(vcost)
            if epoch > 0 and np.isfinite(self.cost_[-1]) and np.isfinite(self.cost_[-2]):
                 if abs(self.cost_[-1] - self.cost_[-2]) < self.tol:
                      print(f"Convergence tolerance reached at epoch {epoch+1}.")
                      break
            elif not np.isfinite(cost):
                 print(f"Cost became non-finite ({cost}) at epoch {epoch+1}. Stopping training.")
                 break


        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise Exception("Model not fitted yet. Call fit() first.")
        Xn = self._normalize_data(X.copy(), fit=False)
        activations, _ = self._forward_pass(Xn)
        y_norm = activations[-1]
        y_pred = self._denormalize_predictions(y_norm)
        return y_pred.reshape(-1) if y_pred.shape[1] == 1 else y_pred

    def score(self, X, y):
        y_true = y.copy()
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = self.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if y_true.shape != y_pred.shape:
             if y_true.shape[0] == y_pred.shape[0] and y_true.shape[1] != y_pred.shape[1]:
                  y_pred = y_pred.reshape(y_true.shape)
             else:
                  raise ValueError(f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape} in score method.")

        mse = np.mean((y_true - y_pred) ** 2)
        return mse 

    def get_confusion_matrix(self, X, y):
        raise NotImplementedError("Confusion matrix is not applicable for regression tasks.")
