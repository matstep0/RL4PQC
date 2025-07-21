import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler  
import argparse

def load_circuit_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    circuit = data.get('circuit')
    dataset = data.get('dataset')
    test_accuracy = data.get('test_accuracy', None)
    return circuit, dataset, test_accuracy

def extract_features_from_circuit(circuit):
    features = set()
    for gate in circuit:
        if gate[0] == 'phased_rx_gate' and gate[1]:
            for param in gate[1]:
                if param.startswith('x_'):
                    features.add(param)
    return sorted(features, key=lambda x: int(x.split('_')[1]))

def get_feature_indices(used_features):
    # Extract integer index from 'x_N' strings
    return [int(f.split('_')[1]) for f in used_features]

def reduce_dataset(X, indices):
    return X[:, indices]

def train_and_evaluate_classical(X, y):
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"Logistic Regression (subset features) CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

def count_circuit_parameters(circuit):
    count = 0
    for gate in circuit:
        if gate[0] == 'phased_rx_gate' and gate[1]:
            # Count all parameters (both x_* and t_*)
            count += len(gate[1])
    return count

def count_trainable_parameters(circuit):
    count = 0
    for gate in circuit:
        if gate[0] == 'phased_rx_gate' and gate[1]:
            # Count only 't_*' parameters
            count += sum(1 for param in gate[1] if param.startswith('t_'))
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare quantum and classical models on circuit features.")
    parser.add_argument("--circuit_json", type=str, required=True, help="Path to circuit JSON file")
    args = parser.parse_args()

    print(f"Loading circuit from {args.circuit_json}")
    # Load circuit and dataset name
    circuit, dataset_name, test_acc = load_circuit_json(args.circuit_json)

    # Load dataset
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    X, y = data.data, data.target
    feature_names = data.feature_names
    print("Feature names:", feature_names)

    # Scale full dataset to [0, pi] for rotation logic
    scaler = MinMaxScaler((0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # Extract used features from circuit
    used_features = extract_features_from_circuit(circuit)
    print(f"Features used in circuit: {used_features}")

    # Map to indices and reduce dataset
    indices = get_feature_indices(used_features)
    print(f"Feature indices: {indices}")
    X_reduced = reduce_dataset(X_scaled, indices)  # Use scaled data for reduction
    print(f"Reduced dataset shape: {X_reduced.shape}")

    # Split data as in rloop.py
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.4, random_state=2138)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    X_train_reduced = reduce_dataset(X_train, indices)
    X_test_reduced = reduce_dataset(X_test, indices)
    print(f"Train shape: {X_train_reduced.shape}, Test shape: {X_test_reduced.shape}")



    # Train logistic regression on train set, test on test set
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_reduced, y_train)
    test_acc = clf.score(X_test_reduced, y_test)
    print(f"Logistic Regression test accuracy: {test_acc:.4f}")

    # Print quantum circuit accuracy if available
    with open(args.circuit_json, 'r') as f:
        data_json = json.load(f)
    if "test_accuracy" in data_json:
        print(f"Quantum circuit accuracy: {data_json['test_accuracy']}")
    else:
        print("Quantum circuit accuracy not found in JSON.")

    # Count number of parameters in the circuit
    num_params = count_circuit_parameters(circuit)
    print(f"Number of parameters in circuit: {num_params}")

    # Count number of trainable parameters in the circuit
    num_trainable_params = count_trainable_parameters(circuit)
    print(f"Number of trainable parameters in circuit: {num_trainable_params}")

    sn_features = X_reduced.shape[1]
    n_classes = len(np.unique(y))
    n_params_logreg = sn_features * n_classes + n_classes
    print(f"Number of trainable parameters in logistic regression: {n_params_logreg}")

    #Retrain and evaluate a quantum circuit from JSON
    #Requires RL4PQC environment and test_circuit_utils.py in PYTHONPATH (probably)
    #Requires setup of rl4pqc package
    import sys
    #sys.path.append('$PWD/.')
    from test_circuit_utils import test_circuit
    test_acc, best_params = test_circuit(args.circuit_json, X_train, y_train, X_test, y_test, lr=0.02, epochs=20, plot=True)
    print(f"Quantum circuit retrained test accuracy: {test_acc}")
    print(f"Corresponding parameters: {best_params}")
