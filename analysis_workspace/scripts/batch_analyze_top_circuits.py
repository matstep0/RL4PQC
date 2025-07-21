import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser

from test_circuit_utils import test_circuit

def extract_features_from_circuit(circuit):
    features = set()
    for gate in circuit:
        if gate[0] == 'phased_rx_gate' and gate[1]:
            for param in gate[1]:
                if param.startswith('x_'):
                    features.add(param)
    return sorted(features, key=lambda x: int(x.split('_')[1]))

def get_feature_indices(used_features):
    return [int(f.split('_')[1]) for f in used_features]

def reduce_dataset(X, indices):
    return X[:, indices]

def count_circuit_parameters(circuit):
    count = 0
    for gate in circuit:
        if gate[1] is not None:
        #if gate[0] == 'phased_rx_gate' and gate[1]:
            count += len(gate[1])
    return count

def count_trainable_parameters(circuit):
    count = 0
    for gate in circuit:
        if gate[1] is not None:

        #if gate[0] == 'phased_rx_gate' and gate[1]:
            count += sum(1 for param in gate[1] if param.startswith('t_'))
    return count

def analyze_circuit(entry, dataset_name, plot_save_path=None):
    circuit = entry['circuit']
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
    print(f"Dataset: {dataset_name}, Features: {feature_names}")
    # Scale full dataset
    scaler = MinMaxScaler((0, np.pi))
    X_scaled = scaler.fit_transform(X)
    # Extract features
    used_features = extract_features_from_circuit(circuit)
    indices = get_feature_indices(used_features)
    X_reduced = reduce_dataset(X_scaled, indices)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.4, random_state=2137) # To use the same split as in rloop.py
    X_train_reduced = reduce_dataset(X_train, indices)
    X_test_reduced = reduce_dataset(X_test, indices)
    # Classical model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_reduced, y_train)
    logreg_acc = float(clf.score(X_test_reduced, y_test))
    # Quantum retrain
    qc_acc, best_params = test_circuit(circuit, X_train, y_train, X_test, y_test, lr=0.005, epochs=2, plot=False, plot_save_path=plot_save_path)
    qc_acc = float(qc_acc)
    # Metadata
    meta = {
        'dataset': dataset_name,
        'validation_accuracy': float(entry.get('validation_accuracy', None)),
        'reward': float(entry.get('reward', 0)),
        'circuit': circuit,
        'log_line': entry.get('log_line'),
        'line_number': int(entry.get('line_number', -1)),
        'used_features': [str(f) for f in used_features],
        'num_all_params': int(count_circuit_parameters(circuit)),
        'num_trainable_params': int(count_trainable_parameters(circuit)),
        'num_logreg_params': int(X_reduced.shape[1] * len(np.unique(y)) + len(np.unique(y))),
        'logreg_accuracy': logreg_acc,
        'quantum_accuracy': qc_acc,
        'best_params': [float(p) for p in best_params] if best_params is not None else None
    }
    return meta

def main():
    parser = ArgumentParser(description="Batch analyze top circuits and compare trainability.")
    parser.add_argument('--top_circuits_json', type=str, required=True, help='Path to top circuits JSON')
    parser.add_argument('--dataset', type=str, required=True, choices=['iris', 'wine', 'breast_cancer'], help='Dataset name')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for summary and plots')
    parser.add_argument('--plot', action='store_true', help='Show trainability plot')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    training_plots_dir = os.path.join(args.output_dir, 'training_plots')
    os.makedirs(training_plots_dir, exist_ok=True)
    summary_json_path = os.path.join(args.output_dir, 'top_circuits_analysis.json')
    summary_plot_path = os.path.join(args.output_dir, 'summary_plot.png')

    with open(args.top_circuits_json, 'r') as f:
        top_circuits = json.load(f)

    results = []
    for entry in top_circuits:
        circuit_id = str(entry.get('line_number', None))
        plot_save_path = os.path.join(training_plots_dir, f'{circuit_id}.png')
        meta = analyze_circuit(entry, args.dataset, plot_save_path=plot_save_path)
        results.append(meta)
        print(f"Circuit {meta['line_number']} | Quantum: {meta['quantum_accuracy']:.4f} | Classical: {meta['logreg_accuracy']:.4f}")
        print(f"Quantum trainable params: {meta['num_trainable_params']} | Logistic Regression params: {meta['num_logreg_params']}")

    with open(summary_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis to {summary_json_path}")

    if args.plot:
        labels = [str(meta['line_number']) for meta in results]
        quantum_acc = [meta['quantum_accuracy'] for meta in results]
        logreg_acc = [meta['logreg_accuracy'] for meta in results]
        quantum_params = [meta['num_trainable_params'] for meta in results]
        logreg_params = [meta['num_logreg_params'] for meta in results]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(12,6))
        bars1 = plt.bar(x - width/2, quantum_acc, width, label='Quantum')
        bars2 = plt.bar(x + width/2, logreg_acc, width, label='Logistic Regression')
        plt.xticks(x, labels, rotation=45)
        plt.ylabel('Test Accuracy')
        plt.xlabel('Circuit ID')
        plt.title('Top PQC vs. logistic regression.')
        plt.legend(loc='lower right', title='p = trainable parameters', framealpha=1)
        plt.tight_layout()
        for i, bar in enumerate(bars1):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{quantum_params[i]}p', ha='center', va='bottom', fontsize=15, color='#1f77b4', fontweight='bold')
        for i, bar in enumerate(bars2):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{logreg_params[i]}p', ha='center', va='bottom', fontsize=15, color='#ff7f0e', fontweight='bold')
        plt.savefig(summary_plot_path)
        print(f"Saved summary plot to {summary_plot_path}")
        plt.show()

if __name__ == '__main__':
    main()
