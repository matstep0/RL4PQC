import json
import numpy as np
import torch
import random
from rl4pqc.pqcgen import RandomPQCGenerator
from rl4pqc.pqceval import PQCEvaluator
from rl4pqc.architectures.helmi5q import HelmiArchitecture
import matplotlib.pyplot as plt

def test_circuit(circuit, X_train, y_train, X_test, y_test, trainable_params=None, lr=0.005, epochs=20, plot=True, seed=None, w_num=5, t_num=20, x_num=None, plot_save_path=None):
    if seed is None:
        seed = random.randint(0, 10000) 
        print(f"Using random seed: {seed}")

    # Use provided w_num and t_num, or defaults
    if x_num is None:
        # Infer x_num from data shape if not provided
        print(X_train.shape)
        x_num = X_train.shape[1]
        print(x_num)
    num_classes = len(np.unique(y_train))
    
    # Setup generator
    pqc_generator = RandomPQCGenerator(w_num=w_num, x_num=x_num, t_num=t_num, seed=seed, architecture=HelmiArchitecture(), max_gates=len(circuit))
    if isinstance(circuit, list):
        pqc_generator.load_circuit(circuit=circuit)
    else:
        pqc_generator.load_circuit(filepath=circuit)
    
    # Setup parameters
    if trainable_params is None:
        trainable_params = np.random.uniform(0, np.pi, size=t_num)
        print(f"Using random trainable parameters: {trainable_params}")
    
    # Setup evaluator
    model = PQCEvaluator(
        n_wires=w_num,
        input_size=x_num,
        params_num=t_num,
        num_classes=num_classes,
        generator=pqc_generator,
        seed=seed+7,
        init_params=trainable_params,
        lr=lr,
        device_name='lightning.qubit',
    )
    model.create_circuit()
    
    # Train and evaluate
    train_losses, val_losses, eval_train, eval_valid, best_params = model.fit(
        train_data=(X_train, y_train),
        valid_data=(X_test, y_test),
        epochs=epochs,
        verbose=True,
    )
    test_acc = np.max(eval_valid)

    # Plot
    if plot:
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure()
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Test Loss')
        plt.plot(epochs_range, eval_train, label='Train Accuracy')
        plt.plot(epochs_range, eval_valid, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(epochs_range)
        plt.title('Training / Validation Curves')
        if plot_save_path:
            plt.savefig(plot_save_path)
            plt.close()
        else:
            plt.show()
    return test_acc, best_params
