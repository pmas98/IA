import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
   
    n_samples_X = X.shape[0]
    n_samples = n_samples_X
    n_test = int(n_samples * test_size)
    
    classes = np.unique(y)
    train_idx = []
    test_idx = []
    
    for c in classes:
        class_idx = np.where(y == c)[0]
        np.random.shuffle(class_idx)
        n_test_class = int(len(class_idx) * test_size)
        test_idx.extend(class_idx[:n_test_class])
        train_idx.extend(class_idx[n_test_class:])
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
   
    X_train = X[train_idx]
    X_test = X[test_idx]
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

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict_classes(X_train)
    y_test_pred  = model.predict_classes(X_test)

    # overall accuracy
    accuracy = accuracy_score(y_test, y_test_pred)

    # get confusion matrix
    cm = calculate_confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2,2)
                        else (None, None, None, None))

    # binary case
    if cm.shape == (2, 2):
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # multiclass: macro-average recall for each class as "sensitivity"
        sensitivity = recall_score(y_test, y_test_pred, average='macro')
        # multiclass "specificity": treat each class as negative, average the negative-class recall
        # sklearn doesn't provide specificity directly, so:
        spec_per_class = []
        for i in range(cm.shape[0]):
            # for class i: negatives = all samples not in class i
            # specificity_i = TN_i / (TN_i + FP_i)
            fp_i = cm[:, i].sum() - cm[i, i]
            tn_i = cm.sum() - cm[i, :].sum() - fp_i
            denom = tn_i + fp_i
            spec_per_class.append(tn_i / denom if denom > 0 else 0.0)
        specificity = np.mean(spec_per_class)

    return {
        'accuracy':      accuracy,
        'sensitivity':   sensitivity,
        'specificity':   specificity,
        'train_accuracy': accuracy_score(y_train, y_train_pred)
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
                         random_seed=None):
    acc_list = []
    sens_list = []
    spec_list = []
    best_model = None
    worst_model = None
    best_score = -np.inf
    worst_score = np.inf

    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
        seeds = rng.randint(0, 10000, size=n_iterations)
    else:
        seeds = np.random.randint(0, 10000, size=n_iterations)

    for _, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(seed)
        )

        model = model_class(**model_params)
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        metrics = model.calculate_metrics(X_test, y_test)

        acc = metrics['accuracy']
        acc_list.append(acc)
        
        if acc > best_score:
            best_score = acc
            best_model = model
        if acc < worst_score:
            worst_score = acc
            worst_model = model
        
        if 'macro_sensitivity' in metrics:
            sens_list.append(metrics['macro_sensitivity'])
            spec_list.append(metrics['macro_specificity'])
        elif 'sensitivity' in metrics:
            sens_list.append(metrics['sensitivity'])
            spec_list.append(metrics['specificity'])
        else:
            sens_list.append(0.0)
            spec_list.append(0.0)

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
        'best_model':         best_model,
        'worst_model':        worst_model
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
    classes = np.unique(y)
    train_idx = []
    val_idx = []
    
    for c in classes:
        class_idx = np.where(y == c)[0]
        np.random.shuffle(class_idx)
        n_val_class = int(len(class_idx) * test_size)
        val_idx.extend(class_idx[:n_val_class])
        train_idx.extend(class_idx[n_val_class:])
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def standard_scale(train, val=None):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_s = (train - mean) / std
    if val is None: return train_s, mean, std
    return train_s, (val - mean) / std, mean, std
