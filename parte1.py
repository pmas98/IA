import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Adeline import Adaline
from Perceptron import Perceptron
from MLP import MLP
from RBF import RBF
from utils import monte_carlo_evaluation, train_test_split
# Definindo cores para os gráficos
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def main():
    # --------------------
    # Exemplo 1: Regressão
    # --------------------
    # Carrega os dados (aerogerador.dat) - ajusta o caminho se necessário
    # data = np.loadtxt("aerogerador.dat")
    # X_reg = data[:, 0].reshape(-1, 1)
    # y_reg = data[:, 1]

    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=X_reg.flatten(), y=y_reg, alpha=0.6)
    # plt.title('Dispersão dos Dados do Aerogerador')
    # plt.xlabel('Velocidade do Vento')
    # plt.ylabel('Potência Gerada')
    # plt.grid(True)
    # plt.show()

    # # Define parâmetros do modelo (SGDRegressor como ADALINE)
    # reg_params = {
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "tol": 1e-5
    # }
    
    # # Avaliação Monte Carlo
    # reg_results = monte_carlo_evaluation(
    #     model_class=Adaline,
    #     model_params=reg_params,
    #     X=X_reg,
    #     y=y_reg,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=False
    # )

    # # Exibe resultados detalhados
    # print("\nResultados da Regressão para Adeline:")
    # print(f"MSE Médio: {reg_results['mean_mse']:.4f} ± {reg_results['std_mse']:.4f}")
    # print(f"MSE Mínimo: {reg_results['min_mse']:.4f}")
    # print(f"MSE Máximo: {reg_results['max_mse']:.4f}")

    # adaline_model = Adaline(learning_rate=0.01, epochs=100, tol=1e-5)
    # adaline_model.fit(X_reg, y_reg)

    # # Predictions from the model
    # y_pred = adaline_model.predict(X_reg)

    # # Plot real data and model predictions on the same scatterplot
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=X_reg.flatten(), y=y_reg, alpha=0.6, label="Dados Reais")
    # sns.scatterplot(x=X_reg.flatten(), y=y_pred, alpha=0.6, label="Previsões do Modelo", color="red")
    # plt.title('Comparação entre Dados Reais e Previsões do Modelo')
    # plt.xlabel('Velocidade do Vento')
    # plt.ylabel('Potência Gerada')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # mlp_reg_params = {
    #     "hidden_layers": (1,),
    #     "activation": 'tanh',
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "tol": 1e-5
    # }

    # # Avaliação Monte Carlo para MLP Underfitting
    # mlp_underfitting_results = monte_carlo_evaluation(
    #     model_class=MLP,
    #     model_params=mlp_reg_params,
    #     X=X_reg,
    #     y=y_reg,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=False
    # )

    # # Exibe resultados detalhados para MLP Underfitting
    # print("\nResultados da Regressão para MLP Underfitting:")
    # print(f"MSE Médio: {mlp_underfitting_results['mean_mse']:.4f} ± {mlp_underfitting_results['std_mse']:.4f}")
    # print(f"MSE Mínimo: {mlp_underfitting_results['min_mse']:.4f}")
    # print(f"MSE Máximo: {mlp_underfitting_results['max_mse']:.4f}")

    # MLP_under = MLP(
    #     hidden_layers=(2,),
    #     activation='tanh',
    #     learning_rate=0.01,
    #     epochs=100,
    #     tol=1e-5
    # )
    # X_train, X_val, y_train, y_val = train_test_split(X_reg, y_reg, test_size=0.2)
    # model = MLP_under.fit(X_train, y_train, X_val, y_val)
    # plt.figure(figsize=(10, 6))
    # plt.plot(model.cost_, label='Treinamento')
    # if model.val_cost_:
    #     plt.plot(model.val_cost_, label='Validação')
    # plt.xlabel('Época')
    # plt.ylabel('Custo (MSE / 2)')
    # plt.title('MLP Curva de Aprendizado - Underfitting')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # mlp_reg_params = {
    #     "hidden_layers": (128, 64, 32),
    #     "activation": 'tanh',
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "tol": 1e-5
    # }

    # # Avaliação Monte Carlo para MLP Underfitting
    # mlp_overfitting_results = monte_carlo_evaluation(
    #     model_class=MLP,
    #     model_params=mlp_reg_params,
    #     X=X_reg,
    #     y=y_reg,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=False
    # )

    # # Exibe resultados detalhados para MLP Underfitting
    # print("\nResultados da Regressão para MLP Overfitting:")
    # print(f"MSE Médio: {mlp_overfitting_results['mean_mse']:.4f} ± {mlp_overfitting_results['std_mse']:.4f}")
    # print(f"MSE Mínimo: {mlp_overfitting_results['min_mse']:.4f}")
    # print(f"MSE Máximo: {mlp_overfitting_results['max_mse']:.4f}")


    # MLP_over = MLP(
    #     hidden_layers=(128, 64, 32),
    #     activation='tanh',
    #     learning_rate=0.01,
    #     epochs=100,
    #     tol=1e-5
    # )
    # model = MLP_over.fit(X_train, y_train, X_val, y_val)
    # plt.figure(figsize=(10, 6))
    # plt.plot(model.cost_, label='Treinamento')
    # if model.val_cost_:
    #     plt.plot(model.val_cost_, label='Validação')
    # plt.xlabel('Época')
    # plt.ylabel('Custo (MSE / 2)')
    # plt.title('MLP Curva de Aprendizado - Overfitting')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # --------------------
    # Exemplo 2: Classificação Spiral3D
    # --------------------
    # Carrega os dados do Spiral3d.csv
    data = np.loadtxt("Spiral3d.csv", delimiter=',')
    X_spiral = data[:, :3]  # Primeiras três colunas como features
    y_spiral = data[:, 3]   # Quarta coluna como target

    # # Visualização 3D dos dados
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(X_spiral[:, 0], X_spiral[:, 1], X_spiral[:, 2], 
    #                     c=y_spiral, cmap='viridis')
    # plt.colorbar(scatter)
    # ax.set_xlabel('Posição eixo X')
    # ax.set_ylabel('Posição eixo Y')
    # ax.set_zlabel('Posição eixo Z')
    # plt.title('Visualização 3D dos Dados Spiral')
    # plt.show()

    # # 1. Perceptron
    # perceptron_params = {
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "tol": 1e-5
    # }

    # perceptron_results = monte_carlo_evaluation(
    #     model_class=Perceptron,
    #     model_params=perceptron_params,
    #     X=X_spiral,
    #     y=y_spiral,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=True
    # )

    # print("\nResultados da Classificação para MLP:")
    # print(f"Acurácia Média: {perceptron_results['mean_accuracy']:.4f} ± {perceptron_results['std_accuracy']:.4f}")
    # print(f"Acurácia Mínima: {perceptron_results['min_accuracy']:.4f}")
    # print(f"Acurácia Máxima: {perceptron_results['max_accuracy']:.4f}")
    # print(f"Sensitividade Média: {perceptron_results['mean_sensitivity']:.4f} ± {perceptron_results['std_sensitivity']:.4f}")
    # print(f"Sensitividade Mínima: {perceptron_results['min_sensitivity']:.4f}")
    # print(f"Sensitividade Máxima: {perceptron_results['max_sensitivity']:.4f}")
    # print(f"Especificidade Média: {perceptron_results['mean_specificity']:.4f} ± {perceptron_results['std_specificity']:.4f}")
    # print(f"Especificidade Mínima: {perceptron_results['min_specificity']:.4f}")
    # print(f"Especificidade Máxima: {perceptron_results['max_specificity']:.4f}")

    # # 2. MLP
    # mlp_params = {
    #     "hidden_layers": (2,),
    #     "activation": 'tanh',
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "task":'classification', 
    # }

    # mlp_results = monte_carlo_evaluation(
    #     model_class=MLP,
    #     model_params=mlp_params,
    #     X=X_spiral,
    #     y=y_spiral,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=True
    # )

    # print("\nResultados da Classificação para MLP:")
    # print(f"Acurácia Média: {mlp_results['mean_accuracy']:.4f} ± {mlp_results['std_accuracy']:.4f}")
    # print(f"Acurácia Mínima: {mlp_results['min_accuracy']:.4f}")
    # print(f"Acurácia Máxima: {mlp_results['max_accuracy']:.4f}")
    # print(f"Sensitividade Média: {mlp_results['mean_sensitivity']:.4f} ± {mlp_results['std_sensitivity']:.4f}")
    # print(f"Sensitividade Mínima: {mlp_results['min_sensitivity']:.4f}")
    # print(f"Sensitividade Máxima: {mlp_results['max_sensitivity']:.4f}")
    # print(f"Especificidade Média: {mlp_results['mean_specificity']:.4f} ± {mlp_results['std_specificity']:.4f}")
    # print(f"Especificidade Mínima: {mlp_results['min_specificity']:.4f}")
    # print(f"Especificidade Máxima: {mlp_results['max_specificity']:.4f}")

    # MLP_under = MLP(
    #     hidden_layers=(2,),
    #     activation='tanh',
    #     learning_rate=0.01,
    #     epochs=100,
    #     task='classification'
    # )


    # X_train, X_val, y_train, y_val = train_test_split(X_spiral, y_spiral, test_size=0.2)
    # model = MLP_under.fit(X_train, y_train, X_val, y_val)

    # accuracy = MLP_under.score(X_val, y_val)

    # # Display results
    # print(f"Validation Accuracy: {accuracy:.4f}\n")
    # metrics = model.calculate_metrics(X_val, y_val)
    # print(f"Validation Sensitivity: {metrics['sensitivity']:.4f}")
    # print(f"Validation Specificity: {metrics['specificity']:.4f}")

    # model.plot_learning_curve()

    # mlp_params = {
    #     "hidden_layers": (128, 64, 32),
    #     "activation": 'tanh',
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "task":'classification', 
    # }

    # mlp_overfitting_results = monte_carlo_evaluation(
    #     model_class=MLP,
    #     model_params=mlp_params,
    #     X=X_spiral,
    #     y=y_spiral,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=True
    # )

    # print("\nResultados da Classificação para MLP:")
    # print(f"Acurácia Média: {mlp_overfitting_results['mean_accuracy']:.4f} ± {mlp_overfitting_results['std_accuracy']:.4f}")
    # print(f"Acurácia Mínima: {mlp_overfitting_results['min_accuracy']:.4f}")
    # print(f"Acurácia Máxima: {mlp_overfitting_results['max_accuracy']:.4f}")
    # print(f"Sensitividade Média: {mlp_overfitting_results['mean_sensitivity']:.4f} ± {mlp_overfitting_results['std_sensitivity']:.4f}")
    # print(f"Sensitividade Mínima: {mlp_overfitting_results['min_sensitivity']:.4f}")
    # print(f"Sensitividade Máxima: {mlp_overfitting_results['max_sensitivity']:.4f}")
    # print(f"Especificidade Média: {mlp_overfitting_results['mean_specificity']:.4f} ± {mlp_overfitting_results['std_specificity']:.4f}")
    # print(f"Especificidade Mínima: {mlp_overfitting_results['min_specificity']:.4f}")
    # print(f"Especificidade Máxima: {mlp_overfitting_results['max_specificity']:.4f}")


    # MLP_classifier = MLP(
    #     hidden_layers=(64, 32, 16),
    #     activation='tanh',
    #     learning_rate=0.01,
    #     epochs=100,
    #     task='classification'  # This is the key addition
    # )

    # X_train, X_val, y_train, y_val = train_test_split(X_spiral, y_spiral, test_size=0.2)
    # model = MLP_classifier.fit(X_train, y_train, X_val, y_val)

    # accuracy = MLP_classifier.score(X_val, y_val)

    # # Display results
    # print(f"Validation Accuracy: {accuracy:.4f}\n")
    # metrics = model.calculate_metrics(X_val, y_val)
    # print(f"Validation Sensitivity: {metrics['sensitivity']:.4f}")
    # print(f"Validation Specificity: {metrics['specificity']:.4f}")

    # model.plot_learning_curve()

    # 3. RBF
    rbf_params = {
        "n_centers": 1,
        "learning_rate": 0.01,
        "epochs": 100,
        "task": 'classification'
    }

    rbf_results = monte_carlo_evaluation(
        model_class=RBF,
        model_params=rbf_params,
        X=X_spiral,
        y=y_spiral,
        n_iterations=100,
        test_size=0.2,
        is_classification=True
    )

    print("\nResultados da Classificação para RBF Under:")
    print(f"Acurácia Média: {rbf_results['mean_accuracy']:.4f} ± {rbf_results['std_accuracy']:.4f}")
    print(f"Acurácia Mínima: {rbf_results['min_accuracy']:.4f}")
    print(f"Acurácia Máxima: {rbf_results['max_accuracy']:.4f}")
    print(f"Sensitividade Média: {rbf_results['mean_sensitivity']:.4f} ± {rbf_results['std_sensitivity']:.4f}")
    print(f"Sensitividade Mínima: {rbf_results['min_sensitivity']:.4f}")
    print(f"Sensitividade Máxima: {rbf_results['max_sensitivity']:.4f}")
    print(f"Especificidade Média: {rbf_results['mean_specificity']:.4f} ± {rbf_results['std_specificity']:.4f}")
    print(f"Especificidade Mínima: {rbf_results['min_specificity']:.4f}")
    print(f"Especificidade Máxima: {rbf_results['max_specificity']:.4f}")
    rbf_results['best_model'].plot_confusion_matrix()
    # 3. RBF
    rbf_params = {
        "n_centers": 25,
        "learning_rate": 0.01,
        "epochs": 100,
        "task": 'classification'
    }

    rbf_results = monte_carlo_evaluation(
        model_class=RBF,
        model_params=rbf_params,
        X=X_spiral,
        y=y_spiral,
        n_iterations=100,
        test_size=0.2,
        is_classification=True
    )

    print("\nResultados da Classificação para RBF Over:")
    print(f"Acurácia Média: {rbf_results['mean_accuracy']:.4f} ± {rbf_results['std_accuracy']:.4f}")
    print(f"Acurácia Mínima: {rbf_results['min_accuracy']:.4f}")
    print(f"Acurácia Máxima: {rbf_results['max_accuracy']:.4f}")
    print(f"Sensitividade Média: {rbf_results['mean_sensitivity']:.4f} ± {rbf_results['std_sensitivity']:.4f}")
    print(f"Sensitividade Mínima: {rbf_results['min_sensitivity']:.4f}")
    print(f"Sensitividade Máxima: {rbf_results['max_sensitivity']:.4f}")
    print(f"Especificidade Média: {rbf_results['mean_specificity']:.4f} ± {rbf_results['std_specificity']:.4f}")
    print(f"Especificidade Mínima: {rbf_results['min_specificity']:.4f}")
    print(f"Especificidade Máxima: {rbf_results['max_specificity']:.4f}")

    # # Plot learning curves for all models
    # plt.figure(figsize=(15, 5))

    # # Perceptron learning curve
    # plt.subplot(131)
    # plt.plot(perceptron_results['best_model'].cost_, label='Training Cost')
    # plt.title('Learning Curve - Perceptron')
    # plt.xlabel('Epochs')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.grid(True)

    # # MLP learning curve
    # plt.subplot(132)
    # plt.plot(mlp_results['best_model'].cost_, label='Training Cost')
    # if hasattr(mlp_results['best_model'], 'val_cost_'):
    #     plt.plot(mlp_results['best_model'].val_cost_, label='Validation Cost')
    # plt.title('Learning Curve - MLP')
    # plt.xlabel('Epochs')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.grid(True)

    # # RBF learning curve
    # plt.subplot(133)
    # plt.plot(rbf_results['best_model'].cost_, label='Training Cost')
    # if hasattr(rbf_results['best_model'], 'val_cost_'):
    #     plt.plot(rbf_results['best_model'].val_cost_, label='Validation Cost')
    # plt.title('Learning Curve - RBF')
    # plt.xlabel('Epochs')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    # # Separa X_raw (N×6) e y_labels (N,)
    # X_raw = np.loadtxt("coluna_vertebral.csv", delimiter=',', usecols=range(6))
    # y_labels = np.loadtxt("coluna_vertebral.csv", delimiter=',', usecols=6, dtype=str)

    # # 2. Organizar X em R^{p×N} e adicionar linha de bias para ficar em R^{(p+1)×N}
    # #    Aqui p = 6, N = número de amostras
    # X = X_raw.T                     # de (N,6) para (6,N)
    # N = X.shape[1]
    # bias = np.ones((1, N))          # linha de 1s
    # X = np.vstack((bias, X))        # agora X é (7, N)

    # # 3. One‐hot encoding em +1/−1 para as 3 classes
    # mapping = {
    #     'NO': np.array([+1, -1, -1]),   # Normal
    #     'DH': np.array([-1, +1, -1]),   # Hérnia de Disco
    #     'SL': np.array([-1, -1, +1])    # Espondilolistese
    # }

    # # Monta Y de forma (3, N): cada coluna i é mapping[y_labels[i]]
    # Y = np.column_stack([mapping[label] for label in y_labels])

    # # Adaline
    # print(f"X shape: {X.shape}")
    # print(f"y shape: {Y.shape}")
    # adaline_params = {
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "tol": 1e-5
    # }
    # adaline_results = monte_carlo_evaluation(
    #     model_class=Adaline,
    #     model_params=adaline_params,
    #     X=X.T,
    #     y=Y.T,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=True
    # )
    # print("\nResultados da Classificação para Adaline:")
    # print(f"Acurácia Média: {adaline_results['mean_accuracy']:.4f} ± {adaline_results['std_accuracy']:.4f}")
    # print(f"Acurácia Mínima: {adaline_results['min_accuracy']:.4f}")
    # print(f"Acurácia Máxima: {adaline_results['max_accuracy']:.4f}")

    # mlp_params = {
    #     "hidden_layers": (10,),
    #     "activation": 'tanh',
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "tol": 1e-5
    # }
    # mlp_results = monte_carlo_evaluation(
    #     model_class=MLP,
    #     model_params=mlp_params,
    #     X=X,
    #     y=Y,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=True
    # )
    # print("\nResultados da Classificação para MLP:")
    # print(f"Acurácia Média: {mlp_results['mean_accuracy']:.4f} ± {mlp_results['std_accuracy']:.4f}")
    # print(f"Acurácia Mínima: {mlp_results['min_accuracy']:.4f}")
    # print(f"Acurácia Máxima: {mlp_results['max_accuracy']:.4f}")

    # rbf_params = {
    #     "n_centers": 10,
    #     "learning_rate": 0.01,
    #     "epochs": 100,
    #     "tol": 1e-5
    # }

    # rbf_results = monte_carlo_evaluation(
    #     model_class=RBF,
    #     model_params=rbf_params,
    #     X=X,
    #     y=Y,
    #     n_iterations=100,
    #     test_size=0.2,
    #     is_classification=True
    # )

    # print("\nResultados da Classificação para RBF:")
    # print(f"Acurácia Média: {rbf_results['mean_accuracy']:.4f} ± {rbf_results['std_accuracy']:.4f}")
    # print(f"Acurácia Mínima: {rbf_results['min_accuracy']:.4f}")
    # print(f"Acurácia Máxima: {rbf_results['max_accuracy']:.4f}")

if __name__ == "__main__":
    main()
