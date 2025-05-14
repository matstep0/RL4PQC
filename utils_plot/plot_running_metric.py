import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_running_metric(
    json_file,
    metric,
    title=None,
    ylabel=None,
    output_prefix=None,
    line_color="red", 
    exploration_phase_points=None,
):
    metric_l = metric.lower()
    data=_get_data(json_file)
    if metric_l in ['max_reward', 'max_rewards']:
        if title is None:
            title = "Maximal Reward Over Episodes"
        if ylabel is None:
            ylabel = "Reward"
        if output_prefix is None:
            output_prefix = "max_reward_"
        # Compute maximum reward per episode.
        episode_val = [max(ep) for ep in data]
        
    elif metric_l in ['accuracy', 'valacc', 'validation_accuracy', 'acc']:
        if title is None:
            title = "Best Validation Accuracy Over Episodes"
        if ylabel is None:
            ylabel = "Accuracy"
        if output_prefix is None:
            output_prefix = "valacc_"
        # Compute maximum accuracy per episode.
        episode_val = [max(ep) for ep in data]
        
    elif metric_l in ['average reward', 'avg_reward', 'avg reward']:
        if title is None:
            title = "Average Reward Over Episodes"
        if ylabel is None:
            ylabel = "Average Reward"
        if output_prefix is None:
            output_prefix = "avg_reward_"
        # Compute average reward per episode.
        episode_val = [sum(ep)/len(ep) if len(ep) > 0 else 0 for ep in data]
    else:
        raise ValueError("Unsupported metric type. Supported types: 'reward', 'accuracy', 'average reward'.")

    # Compute the cumulative best (cumulative maximum) values.
    best_so_far = []
    current_best = float('-inf')
    for value in episode_val:
        current_best = max(current_best, value)
        best_so_far.append(current_best)

    episodes = np.arange(1, len(episode_val) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Add vertical lines for separation of phases
    if exploration_phase_points is not None:
        # Add exploration phase points to the plot
        random=exploration_phase_points.get('random_episodes', None)
        decay=exploration_phase_points.get('decay_episodes', None)
        plt.axvline(x=random + 0.5,
                        color='red',
                        linestyle='--',
                        linewidth=1,
                        alpha=0.7)
        plt.axvline(x=random + decay + 0.5,
                                color='red',
                                linestyle='--',
                                linewidth=1,
                                alpha=0.7) 
       
    # Plot blue scatter for per-episode values (unconnected markers)
    plt.scatter(episodes, episode_val, color='blue', label='Episode Value')
    # Plot continuous line for best-so-far values
    plt.plot(episodes, best_so_far, color=line_color, linestyle='-', linewidth=2, label='Best So Far')

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor= (-0.07, 1.05), fontsize=12) #Fix optimal legend position
    plt.grid(True)

    base_name = os.path.basename(json_file)
    name_without_ext = os.path.splitext(base_name)[0]
    out_path = os.path.join(os.path.dirname(json_file), f"{output_prefix}{name_without_ext}.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")
    plt.close()

def _get_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

