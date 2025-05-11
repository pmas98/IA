import numpy as np

class MLP:
    def __init__(self, hidden_layers=(10,), learning_rate=0.01, epochs=1000,
                 activation='tanh', task='regression'):
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
        self.classes_ = None
        self.X_train = None
        self.y_train = None

    def _initialize_weights(self, n_features, n_outputs):
        layer_sizes = [n_features] + list(self.hidden_layers) + [n_outputs]

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.uniform(-0.5, 0.5, size=(layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _activation_function(self, z, derivative=False):
        if self.activation == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1 - s) if derivative else s
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(z)**2
            return np.tanh(z)

    def _forward_pass(self, X):
        activations = [X]
        z_values = []

        a = X
        for i in range(len(self.weights)):
            z = a.dot(self.weights[i]) + self.biases[i]
            z_values.append(z)

            if i < len(self.weights) - 1:
                a = self._activation_function(z)
            else:
                a = z

            activations.append(a)

        return activations, z_values

    def _compute_cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2) / 2.0

    def _backward_pass(self, X, y, activations, z_values):
        m = X.shape[0]
        n_layers = len(self.weights)

        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y

        dw[-1] = activations[-2].T.dot(delta) / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m

        for l in reversed(range(n_layers - 1)):
            delta = delta.dot(self.weights[l+1].T) * self._activation_function(z_values[l], derivative=True)

            dw[l] = activations[l].T.dot(delta) / m
            db[l] = np.sum(delta, axis=0, keepdims=True) / m

        return dw, db

    def fit(self, X, y, X_val=None, y_val=None, verbose=False, early_stopping=False, patience=20):
        self.X_train = X.copy()

        X_proc = X.copy()
        y_proc = y.copy()

        if y_proc.ndim == 1:
            y_proc = y_proc.reshape(-1, 1)

        self.y_train = y_proc.copy()

        self.classes_ = np.unique(y_proc)
        n_outputs = len(self.classes_)

        if n_outputs == 2:
            y_proc = (y_proc == self.classes_[1]).astype(int)
            n_outputs = 1
        else: 
            label_to_idx = {c: i for i, c in enumerate(self.classes_)}
            idxs = np.array([label_to_idx[c] for c in y_proc.flatten()])
            one_hot = np.zeros((X_proc.shape[0], n_outputs))
            one_hot[np.arange(X_proc.shape[0]), idxs] = 1
            y_proc = one_hot

        X_val_proc, y_val_proc = None, None
        if X_val is not None and y_val is not None:
            X_val_proc = X_val.copy()
            y_val_proc = y_val.copy()

            if y_val_proc.ndim == 1:
                y_val_proc = y_val_proc.reshape(-1, 1)

            if self.task == 'classification':
                if len(self.classes_) == 2:  # Binary
                    y_val_proc = (y_val_proc == self.classes_[1]).astype(int)
                else:
                    val_idxs = np.array([label_to_idx.get(c, -1) for c in y_val_proc.flatten()])
                    
                    val_one_hot = np.zeros((X_val_proc.shape[0], len(self.classes_)))
                    valid_indices = val_idxs != -1
                    val_one_hot[np.arange(X_val_proc.shape[0])[valid_indices], val_idxs[valid_indices]] = 1
                    y_val_proc = val_one_hot

        n_samples, n_features = X_proc.shape

        self._initialize_weights(n_features, n_outputs)

        self.cost_history = []
        self.val_cost_history = []

        best_val_cost = np.inf
        counter = 0
        best_weights = None
        best_biases = None

        for _ in range(self.epochs):
            activations, z_values = self._forward_pass(X_proc)

            cost = self._compute_cost(y_proc, activations[-1])
            self.cost_history.append(cost)

            if X_val_proc is not None and y_val_proc is not None:
                val_activations, _ = self._forward_pass(X_val_proc)
                val_cost = self._compute_cost(y_val_proc, val_activations[-1])
                self.val_cost_history.append(val_cost)
                
                if early_stopping:
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                        counter = 0
                    else:
                        counter += 1

                    if counter >= patience:
                        self.weights = best_weights
                        self.biases = best_biases
                        break

            dw, db = self._backward_pass(X_proc, y_proc, activations, z_values)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dw[i]
                self.biases[i] -= self.learning_rate * db[i]

        self.fitted = True
        return self

    def predict(self, X):
        activations, _ = self._forward_pass(X)
        predictions = activations[-1]
        return predictions

    def predict_classes(self, X):
        predictions = self.predict(X)

        if len(self.classes_) == 2:
            binary_predictions = (predictions >= 0.5).astype(int)
            return self.classes_[binary_predictions.flatten()]
        else:
            class_indices = np.argmax(predictions, axis=1)
            if isinstance(self.classes_, np.ndarray):
                result = self.classes_[class_indices]
            else:
                result = np.array([self.classes_[idx] for idx in class_indices])
            return result

    def score(self, X, y):
        y_pred = self.predict_classes(X)
        return np.mean(y.flatten() == y_pred)

    def plot_learning_curve(self, title='Learning Curve'):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(self.cost_history) + 1)

        # Plot training cost
        plt.plot(epochs_range, self.cost_history, 'b-', label='Treinamento')

        if len(self.val_cost_history) > 0:
            val_epochs_range = range(1, len(self.val_cost_history) + 1)
            plt.plot(val_epochs_range, self.val_cost_history, 'r-', label='Validação')

        plt.title(title)
        plt.xlabel('Épocas')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_confusion_matrix(self, title="Confusion Matrix", label_mapping=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.y_train.ndim > 1 and self.y_train.shape[1] > 1:
            y_true_idx = np.argmax(self.y_train, axis=1)
            y_true_labels = self.classes_[y_true_idx]
        else: 
            y_true_labels = self.y_train.flatten()

        y_pred_labels = self.predict_classes(self.X_train)

        unique_labels = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
        K = len(unique_labels)

        cm = np.zeros((K, K), dtype=int)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        for true_label, pred_label in zip(y_true_labels, y_pred_labels):
            cm[label_to_index[true_label], label_to_index[pred_label]] += 1

        if label_mapping is not None:
            display_labels = [label_mapping.get(label, str(label)) for label in unique_labels]
        else:
            display_labels = [str(label) for label in unique_labels]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels
        )
        plt.title(title)
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self, X, y):
        y_true = y.copy()
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = self.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        y_pred_classes = self.predict_classes(X)
        if y_pred_classes.ndim > 1:
            y_pred_classes = y_pred_classes.flatten()
        y_true_classes = y_true.flatten()

        if len(self.classes_) == 2:
            positive_class = self.classes_[1]
            y_true_binary = (y_true_classes == positive_class).astype(int)
            y_pred_binary = (y_pred_classes == positive_class).astype(int)

            TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1)) # Verdadeiros Positivos
            TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0)) # Verdadeiros Negativos
            FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1)) # Falso Positivos
            FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0)) # Falso Negativos

            total = TP + TN + FP + FN
            accuracy    = (TP + TN) / total if total > 0 else 0
            precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            f1_score    = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'accuracy':    accuracy,
                'precision':   precision,
                'recall':      recall,
                'sensitivity': recall,
                'specificity': specificity,
                'f1_score':    f1_score,
                'confusion_matrix': {
                    'TP': int(TP),
                    'TN': int(TN),
                    'FP': int(FP),
                    'FN': int(FN)
                }
            }
        else:
            K = len(self.classes_)

            cm = np.zeros((K, K), dtype=int)
            idx_map = {c: i for i, c in enumerate(self.classes_)}
            for t, p in zip(y_true_classes, y_pred_classes):
                cm[idx_map[t], idx_map[p]] += 1 

            TP = np.diag(cm).astype(float)
            FP = cm.sum(axis=0) - TP
            FN = cm.sum(axis=1) - TP
            total_samples = cm.sum()
            TN = total_samples - (TP + FP + FN)

            with np.errstate(divide='ignore', invalid='ignore'):
                precision = TP / (TP + FP)
                recall    = TP / (TP + FN)
                f1        = 2 * precision * recall / (precision + recall)
                specificity = TN / (TN + FP)

            precision[np.isnan(precision)] = 0
            recall[np.isnan(recall)] = 0
            f1[np.isnan(f1)] = 0
            specificity[np.isnan(specificity)] = 0

            accuracy = np.trace(cm) / cm.sum()
            macro_precision = precision.mean()
            macro_recall = recall.mean()
            macro_f1 = f1.mean()
            macro_specificity = specificity.mean()

            return {
                'accuracy':           accuracy,
                'precision_per_class': dict(zip(self.classes_, precision)),
                'recall_per_class':    dict(zip(self.classes_, recall)),
                'f1_per_class':        dict(zip(self.classes_, f1)),
                'specificity_per_class': dict(zip(self.classes_, specificity)),
                'macro_precision':     macro_precision,
                'macro_recall':        macro_recall,
                'macro_f1':            macro_f1,
                'sensitivity':         macro_recall,
                'specificity':         macro_specificity,
                'confusion_matrix':    cm
            }
