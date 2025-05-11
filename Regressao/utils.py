import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
   
    n_samples_X = X.shape[0]
    n_samples_y = y.shape[0] if hasattr(y, 'shape') else len(y)
    
    n_samples = n_samples_X
    n_test = int(n_samples * test_size)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
   
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    y_train = y[train_idx]
    y_test = y[test_idx]
   
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_test = np.mean((y_test - y_test_pred) ** 2)
    return {
        'mse_train': mse_train,
        'mse_test': mse_test
    }

def monte_carlo_evaluation(model_class, 
                         model_params,
                         X,         
                         y,         
                         n_iterations=250,
                         test_size=0.2,
                         random_seed=42,
                         y_mean=None,
                         y_std=None):

    mse_list = []
    norm_mse_list = []
    best_model = None
    worst_model = None
    best_score = np.inf
    worst_score = -np.inf

    rng = np.random.RandomState(random_seed)
    seeds = rng.randint(0, 10000, size=n_iterations)

    for _, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(seed)
        )

        y_train_arr = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        y_train_std = y_train_arr.std(axis=0)
        y_train_std[y_train_std == 0] = 1
        y_std_val = float(y_train_std) if np.isscalar(y_train_std) or y_train_std.size == 1 else y_train_std

        if y_mean is not None and y_std is not None:
            model = model_class(**model_params)
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test, y_mean=y_mean, y_std=y_std)
        else:
            model = model_class(**model_params)
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        # Calculate MSE in original scale if scaling params provided
        y_pred = model.predict(X_test)
        
        if y_mean is not None and y_std is not None:
            # Destandardize predictions and true values before computing MSE
            y_test_orig = y_test * y_std + y_mean
            y_pred_orig = y_pred * y_std + y_mean
            orig_mse = np.mean((y_test_orig - y_pred_orig) ** 2)
        else:
            # Use the model's metrics if no scaling factors provided
            metrics = model.calculate_metrics(X_test, y_test)
            orig_mse = metrics['mean_squared_error'] if 'mean_squared_error' in metrics else metrics.get('mse_test', 0.0)
        
        # Calculate normalized MSE for comparison
        norm_mse = orig_mse / (y_std_val ** 2) if y_std_val is not None else orig_mse
        
        mse_list.append(orig_mse)
        norm_mse_list.append(norm_mse)

        if orig_mse < best_score:
            best_score = orig_mse
            best_model = model
        if orig_mse > worst_score:
            worst_score = orig_mse
            worst_model = model

    print("\nEvaluation complete.")

    return {
        'mse_values':        np.array(mse_list),
        'mean_mse':          np.mean(mse_list),
        'std_mse':           np.std(mse_list),
        'min_mse':           np.min(mse_list),
        'max_mse':           np.max(mse_list),
        'norm_mse_values':   np.array(norm_mse_list),
        'mean_norm_mse':     np.mean(norm_mse_list),
        'std_norm_mse':      np.std(norm_mse_list),
        'min_norm_mse':      np.min(norm_mse_list),
        'max_norm_mse':      np.max(norm_mse_list),
        'best_model':        best_model,
        'worst_model':       worst_model
    }

def plot_learning_curves(underfitting_history, overfitting_history):
    plt.figure(figsize=(12, 6))
    
    plt.plot(range(1, len(underfitting_history['train']) + 1), underfitting_history['train'], 
             marker='o', linestyle='-', color='#1f77b4', label='MLP Underfitting (2 neurons) - Training')
    plt.plot(range(1, len(overfitting_history['train']) + 1), overfitting_history['train'], 
             marker='o', linestyle='-', color='#ff7f0e', label='MLP Overfitting (150 neurons) - Training')
    
    if 'val' in underfitting_history:
        plt.plot(range(1, len(underfitting_history['val']) + 1), underfitting_history['val'], 
                 marker='s', linestyle='--', color='#1f77b4', label='MLP Underfitting (2 neurons) - Validation')
    if 'val' in overfitting_history:
        plt.plot(range(1, len(overfitting_history['val']) + 1), overfitting_history['val'], 
                 marker='s', linestyle='--', color='#ff7f0e', label='MLP Overfitting (150 neurons) - Validation')
    
    plt.title('Learning Curves Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_val_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    split = int(X.shape[0] * (1 - test_size))
    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def standard_scale(train, val=None):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_s = (train - mean) / std
    if val is None: return train_s, mean, std
    return train_s, (val - mean) / std, mean, std
