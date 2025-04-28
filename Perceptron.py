import numpy as np

class Perceptron:
    """
    Implementação do modelo Perceptron Simples
    """
    def __init__(self, learning_rate=0.01, epochs=100, random_seed=42, tol=1e-5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.tol = tol
        self.w_ = None  # Pesos a serem aprendidos
        self.train_cost_ = []  # Histórico de custo do treinamento
        self.val_cost_ = []    # Histórico de custo da validação
        self.fitted = False
        self.confusion_matrix = None  # Store confusion matrix
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Ajusta o modelo Perceptron aos dados de treinamento
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados de treinamento
        y : array-like, shape (n_samples,)
            Valores alvo de treinamento
        X_val : array-like, shape (n_val_samples, n_features), optional
            Dados de validação
        y_val : array-like, shape (n_val_samples,), optional
            Valores alvo de validação
            
        Returns:
        --------
        self : object
            Retorna a instância do modelo
        """
        # Criar cópia para não modificar os dados originais
        X_copy = X.copy()
        y_copy = y.copy()
        
        # Adicionar bias como uma coluna de 1s em X
        X_b = np.c_[np.ones((X_copy.shape[0], 1)), X_copy]
        
        # Inicializar pesos
        rng = np.random.RandomState(self.random_seed)
        self.w_ = rng.normal(0.0, 0.05, size=X_b.shape[1])
        
        # Armazenar custos
        self.train_cost_ = []
        self.val_cost_ = []
        
        # Preparar dados de validação se fornecidos
        if X_val is not None and y_val is not None:
            X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
            y_val_copy = y_val.copy()
        
        for i in range(self.epochs):
            errors = 0
            for xi, target in zip(X_b, y_copy):
                # Calcular saída
                output = np.dot(xi, self.w_)
                prediction = np.where(output >= 0.0, 1, -1)
                
                # Atualizar pesos apenas se a previsão estiver errada
                if prediction != target:
                    update = self.learning_rate * (target - prediction)
                    self.w_ += update * xi
                    errors += 1
            
            # Calcular custo de treinamento
            train_cost = errors / len(y_copy)
            self.train_cost_.append(train_cost)
            
            # Calcular custo de validação se dados de validação foram fornecidos
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val_b)
                val_errors = np.sum(val_predictions != y_val_copy)
                val_cost = val_errors / len(y_val_copy)
                self.val_cost_.append(val_cost)
            
            # Verificar convergência
            if errors == 0:
                break
            
            # Verificar convergência usando custo de treinamento
            if i > 0 and abs(self.train_cost_[-1] - self.train_cost_[-2]) < self.tol:
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
    
    def net_input(self, X):
        """Calcula o input da rede"""
        # Adicionar bias se não estiver presente
        if X.shape[1] == self.w_.shape[0] - 1:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X, self.w_)
    
    def predict(self, X):
        """Retorna a classe predita após a função de ativação degrau"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def score(self, X, y):
        """Calcula a acurácia como proporção de amostras classificadas corretamente"""
        return np.mean(self.predict(X) == y)

    def get_confusion_matrix(self, X, y):
        """
        Calculate and return the confusion matrix for the given data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        confusion_matrix : array-like, shape (n_classes, n_classes)
            Confusion matrix where confusion_matrix[i, j] is the number of samples
            with true label i and predicted label j
        """
        if not self.fitted:
            raise Exception("Model not fitted yet")
            
        # Get predictions
        y_pred = self.predict(X)
        
        # Ensure inputs are 1D arrays
        y_true = y.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        # Get number of unique classes
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        
        # Initialize confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        # Fill confusion matrix
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
            
        return cm

