import numpy as np

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
       
    def fit(self, X, y, X_val=None, y_val=None):
        X_copy = X.copy()
        y_copy = y.copy()
       
        X_b = np.c_[np.ones((X_copy.shape[0], 1)), X_copy]
       
        rng = np.random.RandomState(self.random_seed)
        self.w_ = rng.normal(0.0, 0.05, size=X_b.shape[1])
       
        self.train_cost_ = []
        self.val_cost_ = []
       
        if X_val is not None and y_val is not None:
            X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
            y_val_copy = y_val.copy()
       
        n_samples = X_b.shape[0]
       
        for i in range(self.epochs):
            output = np.dot(X_b, self.w_)
           
            errors = y_copy - output
           
            self.w_ += self.learning_rate * np.dot(X_b.T, errors) / n_samples
           
            errors_clipped = np.clip(errors, -1e10, 1e10)
            train_cost = np.mean(errors_clipped**2) / 2.0
            self.train_cost_.append(train_cost)
           
            if X_val is not None and y_val is not None:
                val_output = np.dot(X_val_b, self.w_)
                val_errors = y_val_copy - val_output
                val_errors_clipped = np.clip(val_errors, -1e10, 1e10)
                val_cost = np.mean(val_errors_clipped**2) / 2.0
                self.val_cost_.append(val_cost)
       
        self.fitted = True
        return self
   
    def predict(self, X): 
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X_b, self.w_)
   
    def score(self, X, y):
        predictions = self.predict(X)
        return -np.mean((y - predictions) ** 2)
