import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from utils import decode_individual
from elm import ELM
from geneticalgorithm import geneticalgorithm as ga

# 1. Preparação dos dados
data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_features = X.shape[1]
fitness_history = []

# 2. Função objetivo com ELM
def fitness_elm(individual):
    mask, n_layers, n_neurons, lr = decode_individual(individual, n_features)
    if not any(mask): return 1.0
    X_train_sel = X_train[:, mask]
    X_test_sel = X_test[:, mask]

    clf = ELM(n_hidden_neurons=n_neurons)
    try:
        clf.fit(X_train_sel, y_train)
        preds = clf.predict(X_test_sel)
        acc = np.mean(preds == y_test)
        feature_ratio = sum(mask) / n_features
        fitness = (1 - acc) + 0.1 * feature_ratio
        fitness_history.append(fitness)
        return fitness
    except:
        return 1.0

# 3. Parâmetros do GA
dim = n_features + 3
varbound = np.array([[0, 1]] * n_features +
                    [[1, 3], [5, 50], [0.001, 0.1]])

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

model = ga(function=fitness_elm,
           dimension=dim,
           variable_type='real',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

# 4. Rodar GA
model.run()

# 5. Avaliar melhor solução
best_solution = model.output_dict['variable']
mask, n_layers, n_neurons, lr = decode_individual(best_solution, n_features)

print(f"\nMelhor solução (ELM):\nFeatures: {np.where(mask)[0].tolist()}\nCamadas: {n_layers}, Neurônios: {n_neurons}")

clf_final = ELM(n_hidden_neurons=n_neurons)
clf_final.fit(X_train[:, mask], y_train)
preds = clf_final.predict(X_test[:, mask])
print("\nRelatório de Classificação (ELM):")
print(classification_report(y_test, preds))

# 6. Gráfico de convergência
plt.plot(fitness_history)
plt.xlabel("Avaliação")
plt.ylabel("Fitness")
plt.title("Convergência - GA com ELM")
plt.grid()
plt.savefig("convergencia_ga_elm.png")
