import os
import argparse

from rl4pqc.trainer import AgentTrainer
from utils_plot.plot_running_metric import plot_running_metric

def plot_experiment(folder_path, stamp, filepath_rew, filepath_acc, exploration_phase_points: dict = None):
    #Plot snaphots of rewards throught entire training at given epoches.
    AgentTrainer.plot_rewards(from_file=filepath_rew,
                                to_file=f"{folder_path}/snaplot_{stamp}",
                                num_cols=5,
                                num_rows=4
                               )
    #Plot some metric to measure if we have improvement.
    plot_running_metric(json_file=filepath_rew, metric='max_reward', line_color='red', exploration_phase_points=exploration_phase_points)
    plot_running_metric(json_file=filepath_rew, metric='avg_reward', line_color='violet', exploration_phase_points=exploration_phase_points)
    plot_running_metric(json_file=filepath_acc, metric='accuracy', line_color='green', exploration_phase_points=exploration_phase_points)

def crawl_from_directory(base_dir):
    for current_dir, dirs, files in os.walk(base_dir):
        reward_json = None
        acc_json = None
        
        for fname in files:
            lower_fname = fname.lower()
            if fname.endswith('.json'):
                if 'reward' in lower_fname and reward_json is None:
                    reward_json = os.path.join(current_dir, fname)
                elif ('acc' in lower_fname or 'accuracy' in lower_fname) and acc_json is None:
                    acc_json = os.path.join(current_dir, fname)

        if reward_json and acc_json:
            # Extract the folder’s name as a “stamp”
            stamp = os.path.basename(os.path.normpath(current_dir))
            print(f"Checking {current_dir}, {stamp}")
            plot_experiment(current_dir, stamp, reward_json, acc_json)
            print("\n")

            
def main():
    parser = argparse.ArgumentParser(
        description='Recursively crawl a directory to locate JSON files and make (overwrite) plots.'
    )
    #Positional argument 
    parser.add_argument('root_dir',
                        help='Path to the directory to crawl.')
    args = parser.parse_args()

    start_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(start_dir):
        print(f"Error: '{start_dir}' is not a directory.")
        return

    crawl_from_directory(start_dir)

if __name__ == "__main__":
    main()