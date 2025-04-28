import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
import inspect

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
   
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
   
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    if y.ndim == 2:
        y_train = y[train_idx]
        y_test = y[test_idx]
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]
   
    return X_train, X_test, y_train, y_test

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)

def calculate_confusion_matrix(y_true, y_pred, n_classes=None):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    if n_classes is None:
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def evaluate_model(model, X_train, y_train, X_test, y_test, is_classification=False):    
    if is_classification:
        y_train_pred = model.predict_classes(X_train)
        y_test_pred = model.predict_classes(X_test)

        accuracy = np.mean(y_test_pred == y_test)
        
        cm = calculate_confusion_matrix(y_test, y_test_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = np.mean([
                cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0.0
                for i in range(cm.shape[0])
            ])
            specificity = np.mean([
                (np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])) / 
                (np.sum(cm) - np.sum(cm[i, :])) if (np.sum(cm) - np.sum(cm[i, :])) > 0 else 0.0
                for i in range(cm.shape[0])
            ])
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'train_accuracy': np.mean(y_train_pred == y_train)
        }
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train = np.mean((y_train - y_train_pred) ** 2)
        mse_test = np.mean((y_test - y_test_pred) ** 2)
        
        return {
            'mse_train': mse_train,
            'mse_test': mse_test
        }
    
def plot_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

def monte_carlo_evaluation(model_class, 
                           model_params,
                           X,         
                           y,         
                           n_iterations=250,
                           test_size=0.2,
                           is_classification=False,
                           random_seed=42):

    if is_classification:
        acc_list = []
        sens_list = []
        spec_list = []
    else:
        mse_list = []
        norm_mse_list = []

    rng = np.random.RandomState(random_seed)
    seeds = rng.randint(0, 10000, size=n_iterations)

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(seed)
        )

        if not is_classification:
            y_train_arr = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
            y_train_std = y_train_arr.std(axis=0)
            y_train_std[y_train_std == 0] = 1
            y_std_val = float(y_train_std) if np.isscalar(y_train_std) or y_train_std.size == 1 else y_train_std

        model = model_class(**model_params, random_seed=int(seed))
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        metrics = evaluate_model(
            model,
            X_train, y_train,
            X_test, y_test,
            is_classification=is_classification
        )

        if is_classification:
            acc_list.append(metrics['accuracy'])
            sens_list.append(metrics['sensitivity'])
            spec_list.append(metrics['specificity'])
        else:
            orig_mse = metrics['mse_test']
            norm_mse = orig_mse / (y_std_val ** 2)
            mse_list.append(orig_mse)
            norm_mse_list.append(norm_mse)

    if is_classification:
        return {
            'accuracy_values':    np.array(acc_list),
            'mean_accuracy':      np.mean(acc_list),
            'std_accuracy':       np.std(acc_list),
            'min_accuracy':       np.min(acc_list),
            'max_accuracy':       np.max(acc_list),
            'sensitivity_values': np.array(sens_list),
            'mean_sensitivity':   np.mean(sens_list),
            'std_sensitivity':    np.std(sens_list),
            'min_sensitivity':    np.min(sens_list),
            'max_sensitivity':    np.max(sens_list),
            'specificity_values': np.array(spec_list),
            'mean_specificity':   np.mean(spec_list),
            'std_specificity':    np.std(spec_list),
            'min_specificity':    np.min(spec_list),
            'max_specificity':    np.max(spec_list),
        }
    else:
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