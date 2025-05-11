import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Classificacao.Perceptron import Perceptron
from Classificacao.MLP import MLP
from Classificacao.utils import monte_carlo_evaluation, train_test_split
from Classificacao.Multiclass_Adeline import Adaline as Multiclass_Adaline
# Definindo cores para os gráficos
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def main():
    data = np.loadtxt("Spiral3d.csv", delimiter=',')
    X_spiral = data[:, :3]
    y_spiral = data[:, 3]

    # Visualização 3D dos dados
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_spiral[:, 0], X_spiral[:, 1], X_spiral[:, 2], 
                        c=y_spiral, cmap='viridis')
    plt.colorbar(scatter)
    ax.set_xlabel('Posição eixo X')
    ax.set_ylabel('Posição eixo Y')
    ax.set_zlabel('Posição eixo Z')
    plt.title('Visualização 3D dos Dados Spiral')
    plt.show()

    # 1. Perceptron
    perceptron_params = {
        "learning_rate": 0.1,
        "epochs": 500,
    }

    perceptron_results = monte_carlo_evaluation(
        model_class=Perceptron,
        model_params=perceptron_params,
        X=X_spiral,
        y=y_spiral,
        n_iterations=1,
        test_size=0.2,
    )

    print("\nResultados da Classificação para Perceptron:")
    print(f"Acurácia Média: {perceptron_results['mean_accuracy']:.4f} ± {perceptron_results['std_accuracy']:.4f}")
    print(f"Acurácia Mínima: {perceptron_results['min_accuracy']:.4f}")
    print(f"Acurácia Máxima: {perceptron_results['max_accuracy']:.4f}")
    print(f"Sensitividade Média: {perceptron_results['mean_sensitivity']:.4f} ± {perceptron_results['std_sensitivity']:.4f}")
    print(f"Sensitividade Mínima: {perceptron_results['min_sensitivity']:.4f}")
    print(f"Sensitividade Máxima: {perceptron_results['max_sensitivity']:.4f}")
    print(f"Especificidade Média: {perceptron_results['mean_specificity']:.4f} ± {perceptron_results['std_specificity']:.4f}")
    print(f"Especificidade Mínima: {perceptron_results['min_specificity']:.4f}")
    print(f"Especificidade Máxima: {perceptron_results['max_specificity']:.4f}")

    perceptron_results['best_model'].plot_confusion_matrix("Matriz de Confusão - Melhor Modelo")
    perceptron_results['best_model'].show_decision_boundary()

    # 2. MLP

    mlp_params = {
        "hidden_layers": (64, 32),
        "activation": 'tanh',
        "learning_rate": 0.01,
        "epochs": 500,
        "task":'classification', 
    }

    mlp_ideal_results = monte_carlo_evaluation(
        model_class=MLP,
        model_params=mlp_params,
        X=X_spiral,
        y=y_spiral,
        n_iterations=250,
        test_size=0.2
    )

    print("\nResultados da Classificação para MLP:")
    print(f"Acurácia Média: {mlp_ideal_results['mean_accuracy']:.4f} ± {mlp_ideal_results['std_accuracy']:.4f}")
    print(f"Acurácia Mínima: {mlp_ideal_results['min_accuracy']:.4f}")
    print(f"Acurácia Máxima: {mlp_ideal_results['max_accuracy']:.4f}")
    print(f"Sensitividade Média: {mlp_ideal_results['mean_sensitivity']:.4f} ± {mlp_ideal_results['std_sensitivity']:.4f}")
    print(f"Sensitividade Mínima: {mlp_ideal_results['min_sensitivity']:.4f}")
    print(f"Sensitividade Máxima: {mlp_ideal_results['max_sensitivity']:.4f}")
    print(f"Especificidade Média: {mlp_ideal_results['mean_specificity']:.4f} ± {mlp_ideal_results['std_specificity']:.4f}")
    print(f"Especificidade Mínima: {mlp_ideal_results['min_specificity']:.4f}")
    print(f"Especificidade Máxima: {mlp_ideal_results['max_specificity']:.4f}")

    mlp_ideal_results['best_model'].plot_confusion_matrix("Matriz de Confusão - Overfitting")
    mlp_ideal_results['best_model'].plot_learning_curve("Curva de Aprendizado - Overfitting")
    mlp_ideal_results['worst_model'].plot_confusion_matrix("Matriz de Confusão - Pior Modelo")
    mlp_ideal_results['worst_model'].plot_learning_curve("Curva de Aprendizado - Pior Modelo")

    X_raw = np.loadtxt("coluna_vertebral.csv", delimiter=',', usecols=range(6))
    y_labels = np.loadtxt("coluna_vertebral.csv", delimiter=',', usecols=6, dtype=str)

    X = X_raw.T                     # de (N,6) para (6,N)
    N = X.shape[1]
    bias = np.ones((1, N))          # linha de 1s
    X = np.vstack((bias, X))        # agora X é (7, N)

    mapping = {
        'NO': np.array([+1, -1, -1]),   # Normal
        'DH': np.array([-1, +1, -1]),   # Hérnia de Disco
        'SL': np.array([-1, -1, +1])    # Espondilolistese
    }

    Y = np.column_stack([mapping[label] for label in y_labels])
    label_mapping = {
                'NO': 'Normal',
                'DH': 'Hérnia de Disco', 
                'SL': 'Espondilolistese'
            }

    y_encoded = np.array([mapping[label] for label in y_labels])

    print(f"X shape: {X.shape}")
    print(f"y shape: {Y.shape}")
    adaline_params = {
        "learning_rate": 0.0001,
        "epochs": 500,
        "tol": 1e-6,
    }
    adaline_results = monte_carlo_evaluation(
        model_class=Multiclass_Adaline,
        model_params=adaline_params,
        X=X.T,
        y=y_encoded,
        n_iterations=1,
        test_size=0.2,
    )
    
    print("\nResultados da Classificação para Adaline:")
    print(f"Acurácia Média: {adaline_results['mean_accuracy']:.4f} ± {adaline_results['std_accuracy']:.4f}")
    print(f"Acurácia Mínima: {adaline_results['min_accuracy']:.4f}")
    print(f"Acurácia Máxima: {adaline_results['max_accuracy']:.4f}")
    print(f"Sensitividade Média: {adaline_results['mean_sensitivity']:.4f} ± {adaline_results['std_sensitivity']:.4f}")
    print(f"Sensitividade Mínima: {adaline_results['min_sensitivity']:.4f}")
    print(f"Sensitividade Máxima: {adaline_results['max_sensitivity']:.4f}")
    print(f"Especificidade Média: {adaline_results['mean_specificity']:.4f} ± {adaline_results['std_specificity']:.4f}")
    print(f"Especificidade Mínima: {adaline_results['min_specificity']:.4f}")
    print(f"Especificidade Máxima: {adaline_results['max_specificity']:.4f}")
    adaline_results['best_model'].plot_confusion_matrix("Matriz de Confusão - Melhor Modelo", label_mapping=label_mapping)
    adaline_results['best_model'].plot_learning_curve("Curva de Aprendizado - Melhor Modelo")
    adaline_results['worst_model'].plot_confusion_matrix("Matriz de Confusão - Pior Modelo", label_mapping=label_mapping)
    adaline_results['worst_model'].plot_learning_curve("Curva de Aprendizado - Pior Modelo")
    
    mlp_params = {
        "hidden_layers": (64, 32),
        "activation": 'tanh',
        "learning_rate": 0.02,
        "epochs": 100,
        "task":'classification', 
    }
    mlp_results = monte_carlo_evaluation(
        model_class=MLP,
        model_params=mlp_params,
        X=X.T,
        y=y_labels,
        n_iterations=1,
        test_size=0.2,
    )
    print("\nResultados da Classificação para MLP Overfitting:")
    print(f"Acurácia Média: {mlp_results['mean_accuracy']:.4f} ± {mlp_results['std_accuracy']:.4f}")
    print(f"Acurácia Mínima: {mlp_results['min_accuracy']:.4f}")
    print(f"Acurácia Máxima: {mlp_results['max_accuracy']:.4f}")
    print(f"Sensitividade Média: {mlp_results['mean_sensitivity']:.4f} ± {mlp_results['std_sensitivity']:.4f}")
    print(f"Sensitividade Mínima: {mlp_results['min_sensitivity']:.4f}")
    print(f"Sensitividade Máxima: {mlp_results['max_sensitivity']:.4f}")
    print(f"Especificidade Média: {mlp_results['mean_specificity']:.4f} ± {mlp_results['std_specificity']:.4f}")
    print(f"Especificidade Mínima: {mlp_results['min_specificity']:.4f}")
    print(f"Especificidade Máxima: {mlp_results['max_specificity']:.4f}")
    mlp_results['best_model'].plot_confusion_matrix("Matriz de Confusão - Melhor Modelo", label_mapping=label_mapping)
    mlp_results['best_model'].plot_learning_curve("Curva de Aprendizado - Melhor Modelo")
    mlp_results['worst_model'].plot_confusion_matrix("Matriz de Confusão - Pior Modelo", label_mapping=label_mapping)
    mlp_results['worst_model'].plot_learning_curve("Curva de Aprendizado - Pior Modelo")
    
if __name__ == "__main__":
    main()
