import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w_ = None
        self.errors_ = []
        self.fitted = False
        self.confusion_matrix = None
        self.classes_ = None
        self.label_mapping = None
        self.decision_boundary_data = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None
    
    def fit(self, X, y, X_val=None, y_val=None, feature_names=None):
        self.X_train = X
        self.y_train = y
        self.feature_names = feature_names
        
        if X_val is not None and y_val is not None:
            self.X_val = X_val
            self.y_val = y_val

        self.classes_ = np.unique(y)
            
        self.label_mapping = {
            self.classes_[0]: -1,
            self.classes_[1]: 1
        }
            
        y_copy = np.array([self.label_mapping[label] for label in y])
        
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        self.fitted = True

        for _ in range(self.epochs):
            for xi, target in zip(X, y_copy):
                y_pred = self.predict(xi)
                
                update = self.learning_rate * (target - y_pred)
                
                self.w_[1:] += update * xi
                self.w_[0] += update
        
        self.confusion_matrix = self.get_confusion_matrix(X, y)
        
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def predict_classes(self, X):
        predictions = np.array([self.predict(xi) for xi in X])
        
        return np.array([self.classes_[1] if pred == 1 else self.classes_[0] for pred in predictions])
    
    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)

    def get_confusion_matrix(self, X=None, y=None): 
        if X is None or y is None:
            X = self.X_val
            y = self.y_val
            
        y_pred = self.predict_classes(X)
        
        y_pred_binary = np.array([self.label_mapping[pred] for pred in y_pred])
        y_true_binary = np.array([self.label_mapping[label] for label in y])
        
        cm = np.zeros((2, 2), dtype=int)
        
        cm[0, 0] = np.sum((y_true_binary == -1) & (y_pred_binary == -1))        
        cm[0, 1] = np.sum((y_true_binary == -1) & (y_pred_binary == 1))
        cm[1, 0] = np.sum((y_true_binary == 1) & (y_pred_binary == -1))
        cm[1, 1] = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            
        return cm
    
    def calculate_metrics(self, X, y):
        cm = self.get_confusion_matrix(X, y)
        
        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1_score,
            'confusion_matrix': {
                'TP': int(TP),
                'TN': int(TN),
                'FP': int(FP),
                'FN': int(FN)
            }
        }
    
    def plot_decision_boundary(self):
        X_plot = self.X_train.copy()
        if hasattr(self, 'standardize') and self.standardize and hasattr(self, 'feature_means'):
            X_plot = (X_plot - self.feature_means) / self.feature_stds
        
        h = 0.2
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        z_min, z_max = X_plot[:, 2].min() - 1, X_plot[:, 2].max() + 1
        
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h),
                                np.arange(z_min, z_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        Z = self.net_input(mesh_points)
        Z = Z.reshape(xx.shape)
        
        self.decision_boundary_data = {
            'xx': xx,
            'yy': yy,
            'zz': zz,
            'Z': Z,
            'X_plot': X_plot,
            'y': self.y_train,
            'feature_names': self.feature_names
        }
        
        return self

    def show_decision_boundary(self, title="Fronteira de Decisão"):
        if self.decision_boundary_data is None:
            self.plot_decision_boundary()
            
        import matplotlib.pyplot as plt
        
        X_plot = self.decision_boundary_data['X_plot']
        y = self.decision_boundary_data['y']
        feature_names = self.decision_boundary_data['feature_names']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], 
                            c=y, cmap=plt.cm.RdBu, edgecolors='k', s=40, alpha=0.9)
        
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        
        x_plane = np.linspace(x_min, x_max, 2)
        y_plane = np.linspace(y_min, y_max, 2)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        
        Z_plane = -(self.w_[0] + self.w_[1]*X_plane + self.w_[2]*Y_plane) / self.w_[3]
        
        ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='blue')
        
        if feature_names and len(feature_names) >= 3:
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_zlabel(feature_names[2])
        else:
            ax.set_xlabel('Posição X')
            ax.set_ylabel('Posição Y')
            ax.set_zlabel('Posição Z')
        
        plt.title(title)
        plt.colorbar(scatter, label='Classe')
        
        ax.set_box_aspect([1, 1, 1])
        
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, title="Confusion Matrix"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        labels = ["-1", "1"]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()
        plt.show()
