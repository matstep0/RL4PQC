import os
import argparse
import yaml
import json
import sys
from datetime import datetime

#Data
import numpy as np
import torch 
from sklearn.datasets import make_moons, load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from rl4pqc.pqcgen import RandomPQCGenerator
from rl4pqc.pqceval import PQCEvaluator
from rl4pqc.env import QuantumCircuitEnv
from rl4pqc.agent import DQNAgent
from rl4pqc.trainer import AgentTrainer
from rl4pqc.architectures.helmi5q import HelmiArchitecture


from utils_plot.gen_plots import plot_experiment
import matplotlib.pyplot as plt

#Debug
from icecream import ic
#ic.enable()
ic.disable()

#Profiling
import cProfile
import pstats
use_profiler=False    # Change to True to run profiling   

# Helper function to save the circuit and accuracy as a JSON file
def save_circuit(dir, stamp, circuit_data):
    filename=f"qc_{stamp}"
    output_file = f"{dir}/{filename}.json"
    with open(output_file, 'w') as f:
        json.dump(circuit_data, f, indent=4)
    print(f"Results saved to {output_file}")

# Helper function to generate a filename with dataset and timestamp
def generate_filestamp(dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_name}_{timestamp}"

# Load configuration from a JSON file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
#Data preparation
def load_and_scale_data(dataset_name):
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'digits':
        data = load_digits()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError("Unknown dataset name")

    X, y = data.data, data.target

    # Convert tensor labels to integers
    if hasattr(y[0], 'item'):
        y = np.array([label.item() for label in y])

    # Normalize X to be between 0 and pi
    scaler = MinMaxScaler((0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # Get the number of features and classes
    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    return X_scaled, y, num_features, num_classes

#Dropped at some point but example on how to create generic datasets.
def __prepare_moons(self, data_size, noise):
    X, y = make_moons(n_samples=data_size, noise=noise)
    #y = 2 * y - 1
    scaler = MinMaxScaler((0, np.pi))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, 2, 2

def set_global_seed(seed: int):
    np.random.seed(seed)                   # NumPy global RNG
    torch.manual_seed(seed)                # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # PyTorch GPU (if any)


def plot_losses_and_accuracies(train_losses, val_losses, eval_train, eval_valid, dir, stamp):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Test Loss')
    plt.plot(epochs, eval_train, label='Train Accuracy')
    plt.plot(epochs, eval_valid, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training / Validation Curves')
    plt.show()
    png_path = os.path.join(dir, f"losses_accuracies_{stamp}.png")
    plt.savefig(os.path.join(dir, f"losses_accuracies_{stamp}.png"))
    plt.close()
    print(f"[INFO] Plot saved → {png_path}")

    data = {
        "epochs"        : list(epochs),
        "train_losses"  : [float(v) for v in train_losses],
        "val_losses"    : [float(v) for v in val_losses],
        "train_accuracy": [float(v) for v in eval_train],
        "val_accuracy"  : [float(v) for v in eval_valid],
    }

    json_path = os.path.join(dir, f"losses_accuracies_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Raw data saved → {json_path}")
    


# Main function to load configuration and execute training
def main(config_path, store_directory, agent_weights):
    
    # Load the YAML configuration
    config = load_config(config_path)
    
    # Extract environment and training settings
    env_params = config['environment']
    training_params = config['training']
    agent_params = config['agent']
    
    test_size = config['training']['test_size']
    seed = config['training']['seed']
    set_global_seed(seed)        #! Important for reproducibility
    
    # Setup storage directory
    dataset_name = env_params['dataset']
    dir=store_directory
    try:
        os.makedirs(dir, exist_ok=True)
    #Or switch to user if not available.
    except OSError as e:   
        print(f"[ERROR] Cannot create directory {dir}: {e}")
        print("[INFO] Switching to local directory ~/RL4PQC_Results")
        dir = os.path.expanduser(f"~/RL4PQC_Results/{dir}")
        os.makedirs(dir, exist_ok=True)    
    stamp = generate_filestamp(dataset_name)
    dir = os.path.join(dir, stamp)
    os.makedirs(dir, exist_ok=True)
    
    #Copy config file
    output_file = f"{dir}/config.json"
    with open(output_file, 'w') as f:
        json.dump(config, f, indent = 4)
    
    
    
    # Prepare data
    X_scaled, y, num_features, num_classes = load_and_scale_data(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, stratify=y, random_state=seed)  # stratify for balanced set
    # And keep thet test set for later, it can be big because each data points is simulated - big test size -> faster training, less circuits to execute.
    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    max_samples_per_step = env_params.get('max_samples_per_step', None)
    # Initialize the environment
    env = QuantumCircuitEnv(
        w_num=env_params['w_num'],
        t_num=env_params['t_num'],
        x_num=num_features,
        num_classes=num_classes,
        max_gates=env_params['max_gates'],
        dataset=env_params['dataset'], #Pytroch loarder would be much better here....
        seed=seed+2,   #to be checked if work
        epochs_for_reward=env_params['epochs_for_reward'],
        exploding_reward=env_params['exploding_reward'],  #for testing some idea...
        evaluator_learning_rate=env_params['evaluator_learning_rate'],  # Learning rate for the evaluator model.
        max_samples_per_step=max_samples_per_step,   # Maximum number of samples per step for the environment.
        pl_backend_name=env_params['pl_backend'],
        architecture=HelmiArchitecture(),  # Fetch from config in later dev.
    )   
    valid_size=env_params['validation_size']
    env.load_split_data(X_train, y_train, validation_size=valid_size)  #Evaluation is taken on the validation set.
    env.setup_logging(dir)
    env.log(config)

    # Initialize the agent
    best_buffer_size = int(max(agent_params['buffer_size'],
                           env_params['max_gates'] *
                           training_params['exploration_policy']['random_fraction']*
                           training_params['total_episodes'])) #make it at least random phase size not to lost data
    agent = DQNAgent(
        input_dim = env.observation_space.shape[0] * env.observation_space.shape[1],
        output_dim = env.action_space.n,
        hidden_layers = agent_params['hidden_layers'],
        lr = agent_params['lr'],
        gamma = agent_params['gamma'],
        batch_size = agent_params['batch_size'],
        buffer_size = best_buffer_size,
        target_sync_freq = agent_params['target_sync_freq'],        
        min_replay_size = agent_params['min_replay_size'],
        seed=seed+1,
        device="cpu",
        )
    #Use transfer learning and load previous model.
    print("\n Trying to load parameters")
    try:
        agent.load_weights(agent_weights)
        print(f"Agent weight loaded successfully from {weight_path}")
    except Exception as e:
        print(f"WARNING: Fetched error {e} when trying to load weights. \
              Agent weights were not fetched, initialization will be random. \
              If you intend to train from scrach this is desired behaviour.")


    # Initialize the trainer
    trainer = AgentTrainer(
        env=env,
        agent=agent,
        n_episodes=training_params['total_episodes'],
        max_steps=env_params['max_gates'],       #So we build the circuit sequentially up to max_gates
        exploration_policy=training_params['exploration_policy']
    )
    
    # Print configuration
    print("Configuration data:")
    print("  Environment Parameters:")
    for key, value in env_params.items():
        print(f"    {key}: {value}")
    print("  Training Parameters:")
    for key, value in training_params.items():
        print(f"    {key}: {value}")


    # Can load previous agent here.
    # Juicy training part - RL big loop !!!!!!
    #-------------------------------------------------------------------------------------------------------------------------------------
    print(f"Training {dataset_name} dataset...")
    res = trainer.train()
    agent.save_weights(path=f"{dir}/agent_weights_{stamp}.pt")
    #-------------------------------------------------------------------------------------------------------------------------------------
    
    
    #Print results
    print(f"Results:")
    print(res)
    print(type(res))
    
    
    # Save data
    circuit_data = {
        "dataset": dataset_name,
        "circuit": res['circuit'],
        "accuracy": float(res['accuracy']), # convert to float due to JSON issues
        "params": res['params']
    }
    save_circuit(dir, stamp=stamp, circuit_data=circuit_data)  # save cirquit
    filepath_rew=os.path.join(dir,f"rewards_{stamp}.json")  
    trainer.extract_data(data_type='rewards', to_file=filepath_rew, )  #save rewards
    filepath_acc=os.path.join(dir,f"validation_acc_{stamp}.json")
    trainer.extract_data(data_type='validation_accuracies', to_file=filepath_acc)  # save validation accuracies
    

    #Make plots
    plot_experiment(dir, stamp, filepath_rew, filepath_acc, exploration_phase_points=trainer.get_exploration_points())
    
    #Print acc found
    print(f"RL training finished, best validation acc for {dataset_name}: {circuit_data['accuracy']} \
          For circuit : \
          {res['circuit']},")
    
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Fine tune final circuit to see how good is the architectecture
    #--------------------------------------------------------------------------------------------------------------
    
    #Train also on random parameters? !!!
    #Train the final best circuit(s) on more epoches? Maybe store top X circuits?
    #train_and_evaluate(circuit_data)
    epochs_for_final_training = 20          #ugly harcoded, but shall it be another config file?, will it confuse the user more? 
    lr_for_final_training = 0.02
    #Initialize the generator
    pqc_generator = RandomPQCGenerator(w_num=env_params['w_num'], 
                                            x_num=num_features,
                                            t_num=env_params['t_num'], 
                                            seed=seed+3, 
                                            architecture=HelmiArchitecture(), 
                                            max_gates=env_params['max_gates'],
                                            )
    #Load circuit with generator PQCgen
    
    
    circuit_data_random_params = circuit_data.copy()
    circuit_data_random_params["params"] = np.random.uniform(0, np.pi, size=len(circuit_data["params"])).tolist()
    circuit_data_random_params["accuracy"] = None    

    pqc_generator.load_circuit(circuit=circuit_data['circuit'])  # you can also load a circuit from a file
    
        
    #Initialize the evaluator and optimize the circuits.
    model = PQCEvaluator(
        n_wires=env_params['w_num'],
        input_size=num_features,
        params_num=env_params['t_num'],
        num_classes=num_classes,
        generator=pqc_generator,
        seed=seed+7,                        
        init_params=circuit_data_random_params["params"], #Load params from the best circuit or start with random ones.
        lr=lr_for_final_training,
        device_name=env_params['pl_backend'],   # arguments dictionary for more flexibility ?
    )
    model.create_circuit()
    
    #Evaluate the best circuit on test set
    acc_on_test = model.evaluate(X_test, y_test)
    print(f"Final evaluation on test set: {acc_on_test}")
    
    train_losses, val_losses, eval_train, eval_valid, best_params = model.fit(train_data=(X_train, y_train),  #Train + valid seen by RL
              valid_data=(X_test, y_test),     # Previously separated test set
              epochs=20,
              verbose=True,
            )
    best_acc_on_test_after_finetuning = max(eval_valid)
    plot_losses_and_accuracies(train_losses, val_losses, eval_train, eval_valid, dir, stamp)
    
    print(f"Finetuning finished, best validation acc for {best_acc_on_test_after_finetuning}")
    
    # Save data again (finetuned circuit)
    circuit_data = {
        "dataset": dataset_name,
        "circuit": res['circuit'],
        "test_accuracy": float(best_acc_on_test_after_finetuning), 
        "params": list(best_params)
    }
    save_circuit(dir, stamp=f"{stamp}_finetuned", circuit_data=circuit_data)  # save cirquit
    
    
    print("Goodbye world!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for RL4PQC experiments. Provide configuration file and storage directory.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--store_directory", type=str, required=False, default=".", help="Path to the storage directory.")
    parser.add_argument("--agent_weights", type=str, required=False, default=None, help="Path to pretrained weights")
    args = parser.parse_args()

    #Wrapper would be nice
    if use_profiler: 
        profiler = cProfile.Profile()
        profiler.enable()    
    main(args.config_file, args.store_directory, args.agent_weights)      #Execute main scirpti
    if use_profiler:
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats('cumtime')     # Sort by cumulative time
        stats.print_stats(20)           # Print last 20