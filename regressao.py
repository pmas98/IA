import numpy as np
import matplotlib.pyplot as plt
from Regressao.utils import train_val_split, standard_scale, monte_carlo_evaluation
from Regressao.Adeline import Adaline
from Regressao.MLP_Regressao import MLP

data = np.loadtxt('../aerogerador.dat')
X, y = data[:,0:1], data[:,1]
# split
X_train, X_val, y_train, y_val = train_val_split(X, y)
# scale
X_train_s, X_val_s, x_mean, x_std = standard_scale(X_train, X_val)
y_train_s, y_val_s, y_mean, y_std = standard_scale(y_train.reshape(-1,1), y_val.reshape(-1,1))
y_train_s, y_val_s = y_train_s.flatten(), y_val_s.flatten()

reg_params = {
    "learning_rate": 0.0001,
    "epochs": 1000,
}

reg_results = monte_carlo_evaluation(
    model_class=Adaline,
    model_params=reg_params,
    X=X_train_s,
    y=y_train_s,
    n_iterations=250,
    test_size=0.2,
    y_mean=y_mean,
    y_std=y_std
)

# Exibe resultados detalhados
print("\nResultados da Regressão para Adeline:")
print(f"MSE Médio: {reg_results['mean_mse']:.4f} ± {reg_results['std_mse']:.4f}")
print(f"MSE Mínimo: {reg_results['min_mse']:.4f}")
print(f"MSE Máximo: {reg_results['max_mse']:.4f}")

# # Underfitted model
model_underfitted = MLP(hidden_layers=(16, 8), learning_rate=0.01, epochs=500,
                        activation='tanh', task='regression')
model_underfitted.fit(X_train_s, y_train_s, X_val=X_val_s, y_val=y_val_s, early_stopping=True, y_mean=y_mean, y_std=y_std)
model_underfitted.plot_learning_curve('Curva de Aprendizado MLP - Underfitted')

# Previsão e erro na escala original
X_all_s = (X - x_mean) / x_std
y_pred_s = model_underfitted.predict(X_all_s)
y_pred = y_pred_s * y_std.flatten() + y_mean.flatten()

# Plot
plt.figure(figsize=(8,5))
plt.scatter(X, y, label='Dados Reais', alpha=0.6)
plt.scatter(X, y_pred, label='Predições MLP', alpha=0.6)
plt.xlabel('Velocidade do Vento (m/s)')
plt.ylabel('Potência Gerada')
plt.title('Predição de Energia Eólica - Underfitted')
plt.legend(); plt.grid(True); plt.show()

reg_params = {
    'hidden_layers': (2, ),
    'learning_rate': 0.01,
    'epochs': 500,
    'activation': 'tanh',
    'task': 'regression'
}

reg_results = monte_carlo_evaluation(
    model_class=MLP,
    model_params=reg_params,
    X=X_train_s,
    y=y_train_s,
    n_iterations=1,
    test_size=0.2,
    y_mean=y_mean,
    y_std=y_std
)
print("Monte Carlo Evaluation Results (underfitted):")
print(f"MSE Médio: {reg_results['mean_mse']:.4f} ± {reg_results['std_mse']:.4f}")
print(f"MSE Mínimo: {reg_results['min_mse']:.4f}")
print(f"MSE Máximo: {reg_results['max_mse']:.4f}")

# Overfitted model
model_overfitted = MLP(hidden_layers=(32, 16, 8), learning_rate=0.01,
                        epochs=500, activation='tanh', task='regression')
model_overfitted.fit(X_train_s, y_train_s, X_val=X_val_s, y_val=y_val_s, early_stopping=True, y_mean=y_mean, y_std=y_std)
model_overfitted.plot_learning_curve('Curva de Aprendizado MLP - Overfitted')

# Previsão e erro na escala original (overfitted)
y_pred_s2 = model_overfitted.predict(X_all_s)
y_pred2 = y_pred_s2 * y_std.flatten() + y_mean.flatten()

plt.figure(figsize=(8,5))
plt.scatter(X, y, label='Dados Reais', alpha=0.6)
plt.scatter(X, y_pred2, label='Predições MLP', alpha=0.6)
plt.xlabel('Velocidade do Vento (m/s)')
plt.ylabel('Potência Gerada')
plt.title('Predição de Energia Eólica - Overfitted')
plt.legend(); plt.grid(True); plt.show()

reg_params_large = {
    'hidden_layers': (128, 64, 32),
    'learning_rate': 0.01,
    'epochs': 500,
    'activation': 'tanh',
    'task': 'regression'
}
reg_results2 = monte_carlo_evaluation(
    model_class=MLP,
    model_params=reg_params_large,
    X=X_train_s,
    y=y_train_s,
    n_iterations=1,
    test_size=0.2,
    y_mean=y_mean,
    y_std=y_std
)
print("Monte Carlo Evaluation Results (overfitted):")
print(f"MSE Médio: {reg_results2['mean_mse']:.4f} ± {reg_results2['std_mse']:.4f}")
print(f"MSE Mínimo: {reg_results2['min_mse']:.4f}")
print(f"MSE Máximo: {reg_results2['max_mse']:.4f}")


reg_params_large = {
    'hidden_layers': (32, 16),
    'learning_rate': 0.01,
    'epochs': 200,
    'activation': 'tanh',
    'task': 'regression'
}
reg_results2 = monte_carlo_evaluation(
    model_class=MLP,
    model_params=reg_params_large,
    X=X_train_s,
    y=y_train_s,
    n_iterations=250,
    test_size=0.2,
    y_mean=y_mean,
    y_std=y_std
)
print("Monte Carlo Evaluation Results (final):")
print(f"MSE Médio: {reg_results2['mean_mse']:.4f} ± {reg_results2['std_mse']:.4f}")
print(f"MSE Mínimo: {reg_results2['min_mse']:.4f}")
print(f"MSE Máximo: {reg_results2['max_mse']:.4f}")
