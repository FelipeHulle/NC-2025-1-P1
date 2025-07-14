import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from utils import decode_individual
from scipy.optimize import differential_evolution

# 1. Carregar dados e normalizar
data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_features = X.shape[1]
fitness_history = []

# 2. Função objetivo para DE
def fitness_de(individual):
    mask, n_layers, n_neurons, lr = decode_individual(individual, n_features)
    if not any(mask):
        return 1.0  # solução inválida
    X_train_sel = X_train[:, mask]
    X_test_sel = X_test[:, mask]

    hidden_layers = tuple([n_neurons] * n_layers)
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers,
                        learning_rate_init=lr,
                        max_iter=500,
                        random_state=42)

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

# 3. Limites de busca para DE
bounds = [(0, 1)] * n_features + [       # features binárias (depois arredondadas)
          (1, 3),                         # n_layers
          (5, 50),                        # n_neurons
          (0.001, 0.1)]                   # learning rate

# 4. Rodar DE
result = differential_evolution(fitness_de,
                                 bounds,
                                 strategy='best1bin',
                                 maxiter=30,
                                 popsize=20,
                                 tol=0.01,
                                 mutation=(0.5, 1),
                                 recombination=0.7,
                                 seed=42,
                                 disp=True)

best_solution = result.x
mask, n_layers, n_neurons, lr = decode_individual(best_solution, n_features)

print(f"\nMelhor solução encontrada (DE):")
print(f"Features: {np.where(mask)[0].tolist()}")
print(f"Número de camadas ocultas: {n_layers}")
print(f"Neurônios por camada: {n_neurons}")
print(f"Taxa de aprendizado: {lr:.5f}")

# 5. Avaliar classificador final
X_train_sel = X_train[:, mask]
X_test_sel = X_test[:, mask]

hidden_layers = tuple([n_neurons] * n_layers)
clf_final = MLPClassifier(hidden_layer_sizes=hidden_layers,
                          learning_rate_init=lr,
                          max_iter=500,
                          random_state=42)

clf_final.fit(X_train_sel, y_train)
preds = clf_final.predict(X_test_sel)

print("\nRelatório de classificação (MLP com DE):")
print(classification_report(y_test, preds))

# 6. Gráfico de convergência
plt.plot(fitness_history)
plt.xlabel("Avaliação")
plt.ylabel("Fitness")
plt.title("Convergência - DE com MLP")
plt.grid()
plt.savefig("convergencia_de_mlp.png")
