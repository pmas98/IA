import numpy as np


class RBF:
    """
    Implementação da Rede Neural de Função de Base Radial (RBF)
    """
    def __init__(self, n_centers=10, sigma=1.0, learning_rate=0.01, epochs=100, 
                 batch_size=32, random_seed=42, tol=1e-5):
        self.n_centers = n_centers
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.tol = tol
        self.centers_ = None  # Centros das funções de base radial
        self.weights_ = None  # Pesos da camada de saída
        self.cost_ = []  # Histórico de custo
        self.val_cost_ = []  # Histórico de custo de validação
        self.fitted = False
        self.confusion_matrix = None  # Store confusion matrix

        # Feature scaling parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def _normalize_data(self, X, y=None, fit=False):
        """
        Normalize the input data (and optionally output data) using z-score normalization
        """
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std == 0] = 1  # Avoid division by zero
            
            if y is not None:
                self.y_mean = np.mean(y, axis=0)
                self.y_std = np.std(y, axis=0)
                self.y_std[self.y_std == 0] = 1  # Avoid division by zero
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if y is not None and fit:
            y_norm = (y - self.y_mean) / self.y_std
            return X_norm, y_norm
        elif y is not None:
            y_norm = (y - self.y_mean) / self.y_std
            return X_norm, y_norm
        else:
            return X_norm

    def _denormalize_predictions(self, y_norm):
        """
        Denormalize the predicted values
        """
        return y_norm * self.y_std + self.y_mean
    
    def _rbf_kernel(self, X, centers):
        """
        Calcula a ativação da função de base radial
        
        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados de entrada
        centers : array-like, shape = [n_centers, n_features]
            Centros das funções de base radial
            
        Retorna:
        --------
        rbf_activations : array-like, shape = [n_samples, n_centers]
            Ativações das funções de base radial
        """
        n_samples = X.shape[0]
        n_centers = centers.shape[0]
        rbf_activations = np.zeros((n_samples, n_centers))
        
        for i in range(n_samples):
            for j in range(n_centers):
                # Distância euclidiana entre a amostra e o centro
                dist = np.sum((X[i] - centers[j]) ** 2)
                # Função de ativação gaussiana
                rbf_activations[i, j] = np.exp(-dist / (2 * self.sigma ** 2))
        
        return rbf_activations
    
    def _compute_cost(self, y_true, y_pred):
        """
        Compute mean squared error cost
        """
        m = y_true.shape[0]
        cost = np.sum((y_true - y_pred) ** 2) / (2 * m)
        return cost
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Ajusta o modelo RBF aos dados de treinamento
        
        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados de treinamento
        y : array-like, shape = [n_samples, n_outputs]
            Valores alvo
        X_val : array-like, shape = [n_val_samples, n_features], optional
            Dados de validação
        y_val : array-like, shape = [n_val_samples, n_outputs], optional
            Valores alvo de validação
            
        Retorna:
        --------
        self : objeto
        """
        
        rng = np.random.RandomState(self.random_seed)
        
        # Create copy to avoid modifying original data
        X_copy = X.copy()
        y_copy = y.copy()
        
        # Garantir que y tenha 2 dimensões
        if y_copy.ndim == 1:
            y_copy = y_copy.reshape(-1, 1)
        
        # Apply normalization
        X_norm, y_norm = self._normalize_data(X_copy, y_copy, fit=True)
        
        # Normalize validation data if provided
        if X_val is not None and y_val is not None:
            X_val_copy = X_val.copy()
            y_val_copy = y_val.copy()
            if len(y_val_copy.shape) == 1:
                y_val_copy = y_val_copy.reshape(-1, 1)
            X_val_norm, y_val_norm = self._normalize_data(X_val_copy, y_val_copy)
        
        n_samples, n_features = X_norm.shape
        n_outputs = y_norm.shape[1]
        
        # Selecionar centros aleatoriamente a partir dos dados de entrada
        idx = rng.choice(n_samples, self.n_centers, replace=False)
        self.centers_ = X_norm[idx]
        
        # Inicializar pesos aleatoriamente
        self.weights_ = rng.normal(0, 0.1, (self.n_centers + 1, n_outputs))  # +1 para o bias
        
        self.cost_ = []
        self.val_cost_ = []
        
        # Training parameters
        learning_rate = self.learning_rate
        best_cost = float('inf')
        patience = 10
        patience_counter = 0
        
        # Treinamento do modelo
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_norm[indices]
            y_shuffled = y_norm[indices]
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Calcular ativações RBF
                rbf_activations = self._rbf_kernel(X_batch, self.centers_)
                
                # Adicionar bias
                rbf_with_bias = np.column_stack((np.ones(X_batch.shape[0]), rbf_activations))
                
                # Calcular saída da rede
                output = np.dot(rbf_with_bias, self.weights_)
                
                # Calcular erro e atualizar pesos
                errors = y_batch - output
                delta_w = learning_rate * np.dot(rbf_with_bias.T, errors)
                self.weights_ += delta_w
            
            # Calculate cost on full dataset after each epoch
            rbf_activations = self._rbf_kernel(X_norm, self.centers_)
            rbf_with_bias = np.column_stack((np.ones(n_samples), rbf_activations))
            output = np.dot(rbf_with_bias, self.weights_)
            cost = self._compute_cost(y_norm, output)
            self.cost_.append(cost)
            
            # Calculate validation cost if validation data is provided
            if X_val is not None and y_val is not None:
                val_rbf_activations = self._rbf_kernel(X_val_norm, self.centers_)
                val_rbf_with_bias = np.column_stack((np.ones(X_val_norm.shape[0]), val_rbf_activations))
                val_output = np.dot(val_rbf_with_bias, self.weights_)
                val_cost = self._compute_cost(y_val_norm, val_output)
                self.val_cost_.append(val_cost)
            

            # Learning rate schedule and early stopping
            if cost < best_cost:
                best_cost = cost
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # Reduce learning rate
                    learning_rate *= 0.5
                    patience_counter = 0
                    
                    # If learning rate becomes too small, stop training
                    if learning_rate < 1e-6:
                        break
            
            # Check for convergence
            if epoch > 0 and abs(self.cost_[-1] - self.cost_[-2]) < self.tol:
                break
        

        self.fitted = True
        
        # Update confusion matrix after training
        y_pred = self.predict(X)
        y_true = y.reshape(-1)
        y_pred = y_pred.reshape(-1)
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            self.confusion_matrix[int(t), int(p)] += 1
                
        return self
    
    def predict(self, X):
        """
        Faz previsões utilizando o modelo treinado
        
        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados para previsão
            
        Retorna:
        --------
        y_pred : array-like, shape = [n_samples, n_outputs]
            Previsões
        """
        if not self.fitted:
            raise ValueError("O modelo precisa ser treinado antes de fazer previsões.")
        
        
        # Normalize input
        X_norm = self._normalize_data(X)
        
        # Calcular ativações RBF
        rbf_activations = self._rbf_kernel(X_norm, self.centers_)
        
        # Adicionar bias
        rbf_with_bias = np.column_stack((np.ones(X.shape[0]), rbf_activations))
        
        # Calcular saída da rede
        output = np.dot(rbf_with_bias, self.weights_)
        
        # Denormalize predictions
        predictions = self._denormalize_predictions(output)
        
        return predictions
    
    def score(self, X, y):
        """
        Calcula o erro quadrático médio para regressão ou a acurácia para classificação
        
        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados de teste
        y : array-like, shape = [n_samples, n_outputs]
            Valores alvo
            
        Retorna:
        --------
        score : float
            Erro quadrático médio ou acurácia
        """
        # Garantir que y tenha 2 dimensões
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        y_pred = self.predict(X)
        
        # Caso de regressão (saída única)
        if y.shape[1] == 1:
            return np.mean((y - y_pred) ** 2)  # MSE
        # Caso de classificação (múltiplas saídas)
        else:
            # Para classificação, pegamos o índice com maior valor como classe predita
            y_pred_class = np.argmax(y_pred, axis=1)
            y_true_class = np.argmax(y, axis=1)
            return np.mean(y_pred_class == y_true_class)  # Acurácia
    
    def predict_classes(self, X, threshold=0.5):        
        y_pred = self.predict(X)
        
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:  # Binary classification
            return np.where(y_pred >= threshold, 1, -1)
        else:  # Multi-class: return class with highest score
            return np.argmax(y_pred, axis=1)
