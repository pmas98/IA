import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.001, epochs=100, random_seed=42, tol=1e-5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.tol = tol
        self.w_ = None
        self.cost_history = []
        self.val_cost_history = []
        self.train_acc_ = []
        self.val_acc_ = []
        self.fitted = False
        self.X_train = None
        self.y_train = None
        self.multi_output = False

    def _compute_cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2) / 2.0

    def fit(self, X, y, X_val=None, y_val=None):
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        if X_val is not None and y_val is not None:
            self.X_val = X_val.copy()
            self.y_val = y_val.copy()
        
        X_copy = X.copy()
        y_copy = y.copy()
        

        ones_column = np.ones((X_copy.shape[0], 1))
        
        X_b = np.concatenate([ones_column, X_copy], axis=1)
        
        self.multi_output = y_copy.ndim > 1 and y_copy.shape[1] > 1

        output_dim = y_copy.shape[1] if self.multi_output else 1

        self.w_ = np.zeros((X_b.shape[1], output_dim))
        
        self.cost_history = []
        self.val_cost_history = []
        self.train_acc_ = []
        self.val_acc_ = []
        
        if X_val is not None and y_val is not None:
            X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
            y_val_copy = y_val.copy()
        
        n_samples = X_b.shape[0]
        best_cost = float('inf')
        best_w = None
        
        for i in range(self.epochs):

            output = np.dot(X_b, self.w_)  

            cost = self._compute_cost(y_copy, output)
            self.cost_history.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_w = self.w_.copy()
            
            errors = y_copy - output
            
            gradient = -np.dot(X_b.T, errors) / n_samples
            gradient = np.clip(gradient, -1.0, 1.0)
            self.w_ -= self.learning_rate * gradient
            
            train_pred_class = np.argmax(output, axis=1)
            train_true_class = np.argmax(y_copy, axis=1)
            train_acc = np.mean(train_pred_class == train_true_class)
                
            self.train_acc_.append(train_acc)
            
            if X_val is not None and y_val is not None:
                val_output = np.dot(X_val_b, self.w_)

                val_cost = self._compute_cost(y_val_copy, val_output)
                self.val_cost_history.append(val_cost)

                val_pred_class = np.argmax(val_output, axis=1)
                val_true_class = np.argmax(y_val_copy, axis=1)
                val_acc = np.mean(val_pred_class == val_true_class)

                self.val_acc_.append(val_acc)
            
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tol:
                break
        
        if best_w is not None:
            self.w_ = best_w
            
        self.fitted = True
        return self

    def predict(self, X):
        ones_column = np.ones((X.shape[0], 1))
        X_b = np.concatenate([ones_column, X], axis=1)
        
        return np.dot(X_b, self.w_)

    def predict_classes(self, X):
        out = self.predict(X)
        return np.argmax(out, axis=1)

    def calculate_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_test, axis=1)
        
        accuracy = np.mean(y_pred_class == y_true_class)
        
        n_classes = y_test.shape[1]
        sensitivity = np.zeros(n_classes)
        specificity = np.zeros(n_classes)
        
        for c in range(n_classes):
            tp = np.sum((y_true_class == c) & (y_pred_class == c))
            fp = np.sum((y_true_class != c) & (y_pred_class == c))
            tn = np.sum((y_true_class != c) & (y_pred_class != c))
            fn = np.sum((y_true_class == c) & (y_pred_class != c))
            
            sensitivity[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity[c] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        avg_sensitivity = np.mean(sensitivity)
        avg_specificity = np.mean(specificity)
        
        return {
            'accuracy': accuracy,
            'sensitivity': avg_sensitivity,
            'specificity': avg_specificity
        }

    def score(self, X, y):
        predictions = self.predict(X)
        y_pred_class = np.argmax(predictions, axis=1)
        y_true_class = np.argmax(y, axis=1)
        return np.mean(y_pred_class == y_true_class)

    def plot_learning_curve(self, title='Curva de Aprendizado'):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.cost_history) + 1)
        
        plt.plot(epochs, self.cost_history, 'b-', label='Treinamento')
        if len(self.val_cost_history) > 0:
            plt.plot(epochs, self.val_cost_history, 'r-', label='Validação')
        plt.title(f"{title}")
        plt.xlabel('Épocas')
        plt.ylabel('Custo')
        plt.legend()
        plt.grid(True)
                    
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, title="Confusion Matrix", label_mapping=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        y_pred = self.predict_classes(self.X_val)
        
        if self.multi_output:
            y_true = np.argmax(self.y_val, axis=1)
            
            original_labels = ['NO', 'DH', 'SL']
            y_true_labels = np.array([original_labels[i] for i in y_true])
            y_pred_labels = np.array([original_labels[i] for i in y_pred])
        else:
            y_true_labels = self.y_val.ravel() if self.y_val.ndim > 1 else self.y_val
            y_pred_labels = y_pred
        
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
