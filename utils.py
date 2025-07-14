import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def decode_individual(individual, n_features):
    feature_mask = np.round(individual[:n_features]).astype(bool)
    n_layers = int(round(individual[n_features]))
    n_neurons = int(round(individual[n_features + 1]))
    lr = float(individual[n_features + 2])
    return feature_mask, n_layers, n_neurons, lr

def fitness_function_ga(individual, X_train, y_train, X_test, y_test):
    n_features = X_train.shape[1]
    mask, n_layers, n_neurons, lr = decode_individual(individual, n_features)
    
    if not any(mask):  # evitar vetor vazio
        return 1.0

    X_train_sel = X_train[:, mask]
    X_test_sel = X_test[:, mask]

    hidden_layers = tuple([n_neurons] * n_layers)
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate_init=lr, max_iter=500)

    try:
        clf.fit(X_train_sel, y_train)
        preds = clf.predict(X_test_sel)
        acc = accuracy_score(y_test, preds)
        feature_ratio = sum(mask) / n_features
        return (1 - acc) + 0.1 * feature_ratio
    except:
        return 1.0
