from tqdm import tqdm
import json

import torch
from pennylane import numpy as np

from .env import QuantumCircuitEnv
from .agent import DQNAgent


class AgentTrainer:
    def __init__(self, env, agent, n_episodes=None, max_steps=None, **kwargs):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.random_actions = kwargs.get('random_actions', False)

        # Exploration policy parameters
        self.exploration_policy = kwargs.get('exploration_policy', None)  # None defaults to the agent's policy
        self.policy=self.exploration_policy['type']
        if self.policy == 'epsilon_greedy':
            self.epsilon_start = kwargs.get('epsilon_start', 1.0)
            self.epsilon_end = kwargs.get('epsilon_end', 0.01)
            self.epsilon = self.epsilon_start
            self.agent.set_epsilon(self.epsilon)
        #elif self.policy == 'boltzmann':
            #self.temperature_start = kwargs.get('temperature_start', 1.0) 
            #self.temperature_end = kwargs.get('temperature_end', 0.1)
            #self.temperature = self.temperature_start
        self.random_fraction = kwargs.get('random_fraction', 0.3)  # Fraction of random actions
        self.decay_fraction = kwargs.get('decay_fraction', 0.4)

        # Calculate the episode threshold for exploration phases.
        self.random_episodes = int(self.n_episodes * self.random_fraction)
        self.reuse_sample_after_random_phase = kwargs.get('reuse_sample_after_random_phase', 10)
        self.decay_episodes = int(self.n_episodes * self.decay_fraction)
        self.deterministic_episodes = self.n_episodes - self.random_episodes - self.decay_episodes

        self.training_rewards_lists = []
        self.validation_accuracies_list = []
    
    def get_exploration_points(self):
        return {'random_episodes': self.random_episodes,
                'decay_episodes': self.decay_episodes,
                'deterministic_episodes': self.deterministic_episodes}
    
    def _schedule_exploration(self, episode):
        """Update epsilon or temperature (not implemented yet) based on the current episode."""
        if episode < self.random_episodes:
            self.epsilon = 1.0  # Full exploration
        elif episode <  self.decay_episodes + self.random_episodes:
            # Linear decay of epsilon or temperature
            fraction = (episode - self.random_episodes) / self.decay_episodes
            if self.policy == 'epsilon_greedy': 
                self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
            #elif self.policy == 'boltzmann': 
                #self.temperature = self.temperature_start + fraction * (self.temperature_end - self.temperature_start)
        else:
            self.policy = 'deterministic'  # Switch to deterministic policy after exploration
            
    def _run_episode(self, init_state):
        """Run a single episode building full circuit sequentially."""
        rewards = []
        max_accuracies = []
        max_accuracy = 0 
        best_circuit = []
        best_params = None
        state = init_state # We usually start from empty circuit
    
        for i in range(self.max_steps):
            if self.random_actions:
                action = self.env.action_space.sample()
            elif self.policy == 'epsilon_greedy':
                self.agent.set_epsilon(self.epsilon)
                action = self.agent.select_action(state, policy=self.policy)
            elif self.policy == 'deterministic':
                action = self.agent.select_action(state, policy=self.policy)
            else:
                raise ValueError(f"Unknown policy: {self.policy}")

            next_state, reward, meta = self.env.step(action)
            rewards.append(reward)
            validation_accuracy=meta['max_accuracy']  
            max_accuracies.append(validation_accuracy)

             # Track the best accuracy and corresponding circuit for this episode
            if validation_accuracy > max_accuracy:
                max_accuracy = validation_accuracy
                best_params = meta['best_params']
                best_circuit = self.env.state.copy()
                
            done = False if i < self.max_steps-1 else True
            # Store the transition in the agent's replay buffer
            self.agent.store(state, action, reward, next_state.flatten(), done)
            state = next_state.flatten()

        return rewards, max_accuracies, best_circuit, best_params

    def train(self):
        """Train the agent across multiple episodes."""
        best_max_accuracy = 0
        best_circuit = []
        best_params = None            
        
        for episode in tqdm(range(self.n_episodes), desc="Training Progress"):
            self._schedule_exploration(episode)
            state = self.env.reset().flatten()
            rewards, validation_accuracies, best_qc, best_params = self._run_episode(state)
            max_accuracy=max(validation_accuracies)

            if max_accuracy > best_max_accuracy:
                best_max_accuracy = max_accuracy
                best_circuit = best_qc
                best_params = best_params

            self.training_rewards_lists.append(rewards)
            self.validation_accuracies_list.append(validation_accuracies)
            print(f"Episode {episode}, Total Reward: {np.sum(rewards)}, Best Reward: {np.max(rewards)}, Max Accuracy: {max_accuracy}, Best Params: {best_params}")
            
            # Update the agent's policy after random exploration phase
            if episode == self.random_episodes: #End of random phase - one time bigger update
                reuse = self.reuse_sample_after_random_phase  
                buffer_len = self.agent.replay.__len__()
                offline_updates = int(np.ceil(reuse * buffer_len / self.agent.batch_size))   #Statisticly make each sample be 'reuse' times
                for _ in range(offline_updates):
                    self.agent.update_policy(force_update=True)
            
            # Update the agent's policy each episode
            if episode > self.random_episodes:
                self.agent.update_policy(force_update=False)

        print(f"Best accuracy score: {best_max_accuracy}")
        return {'circuit': best_circuit,
                'accuracy': best_max_accuracy,
                'params': best_params}

    def extract_data(self, data_type : str = None, to_file=None):
        """
        Extract specified data (rewards or validation accuracies) and save to a file.

        Args:
            data_type (str): The type of data to extract ('rewards' or 'validation_accuracies').
            to_file (str): The file path to save the extracted data.
        """
        if data_type == 'rewards':
            data = [[float(r) for r in episode] for episode in self.training_rewards_lists]
        elif data_type == 'validation_accuracies':
            data = [[float(acc) for acc in acc_list] for acc_list in self.validation_accuracies_list]
        else:
            raise ValueError("Invalid data_type. Choose 'rewards' or 'validation_accuracies'.")

        if to_file is not None:
            with open(to_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"{data_type.capitalize()} saved to {to_file}")
        else:
            print(f"No file path provided. {data_type.capitalize()} not saved.")
        

    @staticmethod
    def plot_rewards(from_file=None,
                     to_file=None,
                     num_cols=5,
                     num_rows=4,
                     show=False):
        try:
            import matplotlib.pyplot as plt
            import os
        except Exception as e:
            print(e)            
            return

        # Try loading rewards from file if specified
        if from_file is not None:
            try:
                with open(from_file, 'r') as f:
                    rewards_data = json.load(f)
            except Exception as e:
                print(f"Failed to load reward data from file: {e}")
                return
        else:
            print("No reward data filename provided.")
            return

        #Plan the plotting
        num_episodes = len(rewards_data)
        plot_num = num_cols * num_rows
        interval = max(1, num_episodes // plot_num)

        # Select episodes to plot at regular intervals
        filtered_data = [rewards_data[i] for i in range(num_episodes) if i % interval == 0][:plot_num]
        # Create plot layout based on specified columns and rows
        fig_width = num_cols * 6
        fig_height = num_rows * 3
        plt.figure(figsize=(fig_width, fig_height))

        for idx, episode_rewards in enumerate(filtered_data):
            ax = plt.subplot(num_rows, num_cols, idx+1)
            ax.plot(episode_rewards, marker='o', linestyle='-')
            ax.set_title(f'Episode {idx * interval}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.grid(True)

        plt.tight_layout()
        if show: plt.show()

        plot_filepath = to_file
        plt.savefig(plot_filepath)
        print(f"Figure saved to {plot_filepath}")


"""Example use
env = QuantumCircuitEnv()
agent = DQNAgent(input_dim=40, output_dim=2)
trainer = AgentTrainer(env, agent)
trainer.train()
trainer.plot_rewards()
"""