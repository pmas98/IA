import numpy as np
import matplotlib.pyplot as plt

class Adaline:

    def __init__(self, learning_rate=0.01, epochs=100, random_seed=42, tol=1e-5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.tol = tol
        self.w_ = None
        self.train_cost_ = []
        self.val_cost_ = []
        self.fitted = False
        self.y_mean_ = None
        self.y_std_ = None

    def _compute_cost(self, y_true, y_pred):
        if self.y_mean_ is not None and self.y_std_ is not None:
            y_true_orig = y_true * self.y_std_ + self.y_mean_
            y_pred_orig = y_pred * self.y_std_ + self.y_mean_
            return np.mean((y_true_orig - y_pred_orig) ** 2)
        return np.mean((y_true - y_pred) ** 2) / 2.0

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, patience=10, y_mean=None, y_std=None):
        if y_mean is not None and y_std is not None:
            self.y_mean_ = y_mean
            self.y_std_ = y_std
        else:
            self.y_mean_ = np.mean(y)
            self.y_std_ = np.std(y)
        num_samples = X.shape[0]
        ones_column = np.ones((num_samples, 1))
        X_b = np.concatenate([ones_column, X], axis=1)
        y_copy = y.copy()

        if X_val is not None and y_val is not None:
            num_val_samples = X_val.shape[0]
            ones_column_val = np.ones((num_val_samples, 1))
            X_val_b = np.concatenate([ones_column_val, X_val], axis=1)
            y_val_copy = y_val.copy()

        self.w_ = np.zeros(X_b.shape[1])

        self.train_cost_.clear()
        self.val_cost_.clear()

        best_val = np.inf
        best_w = None
        wait = 0

        n_samples = X_b.shape[0]

        for epoch in range(1, self.epochs + 1):
            output = X_b.dot(self.w_)
            errors = y_copy - output
            self.w_ += self.learning_rate * (X_b.T.dot(errors) / n_samples)
            cost = self._compute_cost(y_copy, output)
            self.train_cost_.append(cost)

            if X_val is not None and y_val is not None:
                val_output = X_val_b.dot(self.w_)
                val_cost = self._compute_cost(y_val_copy, val_output)
                self.val_cost_.append(val_cost)

                if early_stopping:
                    if val_cost < best_val:
                        best_val = val_cost
                        best_w = self.w_.copy()
                        wait = 0 
                    else:
                        wait += 1 
                    if wait >= patience:
                        self.w_ = best_w
                        self.train_cost_ = self.train_cost_[:epoch-wait]
                        self.val_cost_ = self.val_cost_[:epoch-wait]
                        break

        self.fitted = True
        return self

    def predict(self, X):
        quantidade_amostras = X.shape[0]
        coluna_uns = np.ones((quantidade_amostras, 1))
        X_b = np.concatenate([coluna_uns, X], axis=1)
        return X_b.dot(self.w_)

    def plot_learning_curve(self, title='Curva de Aprendizado'):
        plt.figure(figsize=(8, 5))
        epochs_range = np.arange(1, len(self.train_cost_) + 1)
        plt.plot(epochs_range, self.train_cost_, label='Treino')
        if self.val_cost_:
            val_epochs_range = np.arange(1, len(self.val_cost_) + 1)
            plt.plot(val_epochs_range, self.val_cost_, label='Validação')
        plt.title(title)
        plt.xlabel('Épocas')
        plt.ylabel('MSE (Custo)')
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

        y_true_orig = y_true * self.y_std_ + self.y_mean_
        y_pred_orig = y_pred * self.y_std_ + self.y_mean_

        errors = y_true_orig - y_pred_orig

        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))

        return {
            'mean_absolute_error': mae,
            'mean_squared_error': mse,
            'root_mean_squared_error': rmse
        }