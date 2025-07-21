# Standard
import os
from tqdm import tqdm
import json, csv
from pprint import pformat
import logging
from icecream import ic #Debug
#ic.enable()
ic.disable()

import gym
from gym import spaces
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


from rl4pqc.pqcgen import RandomPQCGenerator
from rl4pqc.pqceval import PQCEvaluator
from rl4pqc.architectures.helmi5q import HelmiArchitecture

# Initialize logging
#logging.basicConfig(filename='RF_quantum_circuit_gym.log', level=logging.INFO)


# Custom Gym Environment
#Gym.env compatibility allows to run external agent (not implemented).
class QuantumCircuitEnv(gym.Env):
    def __init__(self,
                 w_num=None, # Wires number
                 t_num=None, # Trainable parameters number
                 x_num=None, # Features number
                 num_classes=None, # Classes number
                 max_gates=None,
                 seed=None,
                 dataset=None,
                 x_chance=0.5,
                 architecture=None,
                 epochs_for_reward=None,
                 exploding_reward=False,
                 max_samples_per_step=None,
                 evaluator_learning_rate=None,
                 pl_backend_name=None):
        super(QuantumCircuitEnv, self).__init__()
        self.seed = seed if seed is not None else np.random.randint(0, 2137)

        # Initialize variables
        self.w_num = w_num
        self.t_num = t_num
        self.x_num = x_num 
        self.num_classes = num_classes
        self.max_gates = max_gates  # Maximal number of gates
        self.x_chance = x_chance # when addint trainable od data(x) parameters
        self.epochs_for_reward = epochs_for_reward
        self.exploding_reward = exploding_reward
        self.evaluator_learning_rate=evaluator_learning_rate
        self.pl_backend_name=pl_backend_name
        self.architecture = architecture
        self.gate_type_map = architecture.get_gate_encoding() if architecture is not None else None
        self.max_samples_per_step = max_samples_per_step    #Set maximum for smaples used in in eval and train due to extensive time.

        self.random_pqc_generator = RandomPQCGenerator(w_num=self.w_num, 
                                                       x_num=self.x_num,
                                                       t_num=self.t_num, 
                                                       seed=self.seed, 
                                                       x_chance=0.5,
                                                       architecture=self.architecture, #If none would load helmi5q as default
                                                       max_gates=self.max_gates,
                                                       )
                                                       
        self.state = self.random_pqc_generator.to_list()
        self.state_box = self.state_to_box(self.state)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.random_pqc_generator.num_actions)

        self.observation_space = spaces.Box(low=0, high=max(
             self.w_num,5), shape=(max_gates, 3), dtype=int)
        #This is special representation observation space inspired by M.Ostaszewski and further modfied, 
        # and may be bottleneck 

        # custom
        self.verbose = False  # make QuantumClassifier silent

    # AFTER INIT DO: data_load, logging and setup
    def load_split_data(self, X: torch.Tensor, Y: torch.Tensor, validation_size: float = 0.3):
        """
        Load data from a PyTorch tensor.
        """
        self.data_features = X
        self.data_labels = Y
        
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.data_features, self.data_labels, test_size=validation_size, random_state=self.seed
        )
        return 
        
    def setup_logging(self, store_directory, log_filename="RF_quantum_circuit_gym.log"):
        os.makedirs(store_directory, exist_ok=True) # Ensure the directory exists
        log_filepath = os.path.join(store_directory, log_filename)
        logging.basicConfig(
            filename=log_filepath,
            filemode='w',  # Overwrite existing logs for a fresh start
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Log the directory setup
        logging.info(f"Logging initialized. Logs will be saved to {log_filepath}")
        print(f"Logs will be saved to {log_filepath}")
        
    def log(self, obj):
        logging.info(pformat(obj))
        print(pformat(obj))

    #API :Next two functions are crucial (API) for any integration with outsource agent loops but need to be adopted.
    def reset(self):
        self.random_pqc_generator.reset()
        #self.random_pqc_generator = RandomPQCGenerator(
            #w_num=self.w_num, x_num=self.x_num, t_num=self.t_num)
        self.state = self.random_pqc_generator.to_list()
        ic(f"Initial state: {self.state}")
        logging.info(f"RESET")
        logging.info(f"Initial state: {self.state}")  # Offline logging

        self.state_box = self.state_to_box(self.state)
        return self.state_box

    def step(self, action):
        #assert action in [0, 1], "Invalid action"
        self.__apply_action(action)
        self.__update_state()

        reward, max_accuracy, best_params = self.__calculate_reward()
        # Offline logging
        logging.info(f"Acc/Rew/St: {max_accuracy} / {reward} / {self.state} ")
        ic(reward)

        return self.state_box, reward, {'max_accuracy': max_accuracy, 'best_params': best_params}


    # Utils
    def __apply_action(self, action_index):
        self.random_pqc_generator.invoke_action(action_index)

    def __update_state(self):
        self.state = self.random_pqc_generator.to_list()
        print(self.state)
        self.state_box = self.state_to_box(self.state)

    def state_to_box(self, state):
        """Implement box representation of a quantum circuit suitable for NN. 
        #Proof of concept is for representing circuits as boxes https://arxiv.org/pdf/2103.16089 """
        # Representation is ambiguous (0,0,0 actually represent a gate.. and lame cause (only gate type and one-qubit is )
        ic(state)
        box_state = np.zeros((self.max_gates, 3), dtype=int)  #each column represent a gate 
        # 0 - gate type as int, 1 - control qubit, 2 - target qubit
        for i, gate in enumerate(state):
            ic(gate)
            box_state[i, 0] = self.gate_type_map.get(gate[0]) # fill the gate type

            if len(gate[2]) > 1:      #represent wiring                     
                [control, target] = gate[2]
            else:
                [control, target] = gate[2][0] , gate[2][0] 
            box_state[i, 1] = control      # fill the qubits used
            box_state[i, 2] = target

        return box_state


    #EVALUATE: Train quantum model
    def __train_model(self):
        # Consider using old already trained params loaded to model.
        model = PQCEvaluator(n_wires=self.w_num,
                                  input_size=self.x_num,
                                  params_num=self.t_num,
                                  num_classes=self.num_classes,
                                  generator=self.random_pqc_generator,
                                  seed=self.seed,
                                  lr=self.evaluator_learning_rate,
                                  device_name=self.pl_backend_name   # arguments dictionary for more flexibilty ?
                                  )
        # model.load_params(best_params) # Can load best params from previous training to improve search 
        self.seed=self.seed+1
        model.create_circuit()
        
        X_train = self.X_train; y_train = self.y_train
        X_valid = self.X_valid; y_valid = self.y_valid
        
        # optional subsampling for efficienty
        if self.max_samples_per_step is not None:
            rng = np.random.default_rng(self.seed) 

            # Sub-sample training set
            if self.max_samples_per_step < len(X_train):
                idx_train = rng.choice(len(X_train), size=self.max_samples_per_step, replace=False)
                idx_train = torch.as_tensor(idx_train, dtype=torch.long)
                X_tr, y_tr = X_train[idx_train], y_train[idx_train]
            else:
                X_tr, y_tr = X_train, y_train

            # Sub-sample validation set
            if self.max_samples_per_step < len(X_valid):
                idx_val = rng.choice(len(X_valid), size=self.max_samples_per_step, replace=False)
                idx_val = torch.as_tensor(idx_val, dtype=torch.long)
                X_val, y_val = X_valid[idx_val], y_valid[idx_val]
            else:
                X_val, y_val = X_valid, y_valid
        else:
            # No subsampling – use full sets
            X_tr,  y_tr  = X_train,  y_train
            X_val, y_val = X_valid, y_valid

        # ---------------- model training ----------------
        train_losses, val_losses, eval_train, eval_valid, best_params = model.fit(
            train_data=(X_tr, y_tr),
            valid_data=(X_val, y_val),
            epochs=self.epochs_for_reward,
            verbose=self.verbose,
        )
        
        return eval_valid, best_params 

    def __calculate_reward(self):
        test_accs, best_params = self.__train_model()
        max_accuracy = np.max(test_accs)
        ic(max_accuracy)
        best_epoch = np.argmax(test_accs)
        ic(f"Max accuracy: {max_accuracy} at best epoch {best_epoch}")
        # This will help check training is stable.
        logging.info(
            f"Max accuracy: {max_accuracy} at best epoch {best_epoch}")
        reward = max_accuracy - 1/self.num_classes  # -0.1 * len(state) to shorter the states 
        # This may not encorage agent to reach for high accuracy which is harder to get there, consider reward exploding at acc 1...
        #Exploding reward function may require another hyperparameter and analysis
        if self.exploding_reward: #FALSE by default
            reward += 1/(1-max_accuracy+1e-6) - 1/(1-1/self.num_classes) # 0 for random acc expload for high acc
        
        return reward, max_accuracy, best_params
