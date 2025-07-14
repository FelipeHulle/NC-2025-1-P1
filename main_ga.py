import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from utils import decode_individual, fitness_function_ga
from geneticalgorithm import geneticalgorithm as ga

# 1. Carregar e preparar os dados
data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_features = X.shape[1]

# 2. Função objetivo (fitness) para o GA
def objective_function(individual):
    return fitness_function_ga(individual, X_train, y_train, X_test, y_test)

# 3. Configuração dos parâmetros do GA
dim = n_features + 3  # [features binárias] + [n_layers, n_neurons, learning_rate]

varbound = np.array([[0, 1]] * n_features +        # seleção de features
                    [[1, 3],                       # n_layers
                     [5, 50],                      # n_neurons
                     [0.001, 0.1]])                # learning rate

algorithm_param = {
    'max_num_iteration': 30,
    'population_size': 20,
    'mutation_probability': 0.1,
    'elit_ratio': 0.1,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

model = ga(function=objective_function,
           dimension=dim,
           variable_type='real',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

# 4. Executa o algoritmo genético
model.run()

# 5. Avalia a melhor solução encontrada
best_solution = model.output_dict['variable']
print("\nBest solution found (raw):")
print(best_solution)

# 6. Decodifica os dados do melhor indivíduo
feature_mask, n_layers, n_neurons, lr = decode_individual(best_solution, n_features)
print(f"\nConfiguração final:")
print(f"- Features selecionadas: {np.where(feature_mask)[0].tolist()}")
print(f"- Número de camadas ocultas: {n_layers}")
print(f"- Neurônios por camada: {n_neurons}")
print(f"- Taxa de aprendizado: {lr:.5f}")

# 7. Treina e avalia o classificador final
X_train_sel = X_train[:, feature_mask]
X_test_sel = X_test[:, feature_mask]
hidden_layers = tuple([n_neurons] * n_layers)

clf = MLPClassifier(hidden_layer_sizes=hidden_layers,
                    learning_rate_init=lr,
                    max_iter=500, random_state=42)

clf.fit(X_train_sel, y_train)
preds = clf.predict(X_test_sel)

# 8. Relatório de métricas
print("\nRelatório de classificação (dados de teste):")
print(classification_report(y_test, preds))
