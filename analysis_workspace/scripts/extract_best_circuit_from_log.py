import argparse
import ast
import json

# Example usage: python extract_best_circuit_from_log.py --log_file path/to/RLlog.log --output_json topN.json

def parse_log_for_best_circuit(log_path):
    best_reward = float('-inf')
    best_circuit = None
    best_acc = None
    best_line = None
    with open(log_path, 'r') as f:
        for line in f:
            if 'Acc/Rew/St:' in line:
                try:
                    # Example line: ... Acc/Rew/St: 0.44999998807907104 / 0.11666664481163025 / [[...]]
                    parts = line.split('Acc/Rew/St:')[1].strip().split(' / ')
                    acc = float(parts[0])
                    reward = float(parts[1])
                    circuit_str = parts[2].strip()
                    circuit = ast.literal_eval(circuit_str)
                    if reward > best_reward:
                        best_reward = reward
                        best_circuit = circuit
                        best_acc = acc
                        best_line = line.strip()
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}\n{e}")
    return best_acc, best_reward, best_circuit, best_line

def parse_log_for_top_circuits(log_path, top_n=1, prefer_latter=False):
    circuits = []
    with open(log_path, 'r') as f:
        for idx, line in enumerate(f, 1):  # idx is line/epoch number, starting from 1
            if 'Acc/Rew/St:' in line:
                try:
                    parts = line.split('Acc/Rew/St:')[1].strip().split(' / ')
                    val_acc = float(parts[0])
                    reward = float(parts[1])
                    circuit_str = parts[2].strip()
                    circuit = ast.literal_eval(circuit_str)
                    circuits.append({
                        'validation_accuracy': val_acc,
                        'reward': reward,
                        'circuit': circuit,
                        'log_line': line.strip(),
                        'line_number': idx
                    })
                except Exception as e:
                    print(f"Error parsing line {idx}: {line.strip()}\n{e}")
    # Sort by reward descending and keep top_n
    if prefer_latter:
        circuits_sorted = sorted(
            circuits,
            key=lambda x: (x['reward'], x['line_number']), # to keep the latest circuit (done by agent) with the same reward
            reverse=True
        )[:top_n]
    else: # just top by reward
        circuits_sorted = sorted(circuits, key=lambda x: x['reward'], reverse=True)[:top_n]
    return circuits_sorted

def main():
    parser = argparse.ArgumentParser(description="Extract top N circuits by reward from RL4PQC log file.")
    parser.add_argument('--log_file', type=str, required=True, help='Path to log file')
    parser.add_argument('--output_json', type=str, help='Optional: path to save top circuits as JSON')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top circuits to keep')
    parser.add_argument('--prefer_latter', action='store_true', default=False, help='Prefer circuits with the same reward that were done later in the log')
    args = parser.parse_args()

    top_circuits = parse_log_for_top_circuits(args.log_file, top_n=args.top_n, prefer_latter=args.prefer_latter)
    
    top_to_print = top_circuits[:10]
    if top_to_print:
        print(f"Top 10 circuits by reward:")
        for i, entry in enumerate(top_to_print, 1):
            print(f"Rank {i} (Line {entry['line_number']}):")
            print(f"  Validation Accuracy: {entry['validation_accuracy']}")
            print(f"  Reward: {entry['reward']}")
            print(f"  Circuit: {entry['circuit']}")
            print(f"  Log line: {entry['log_line']}")
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(top_circuits, f, indent=2)
            print(f"All top circuits dumped to: {args.output_json}")
    else:
        print("No circuit found in log file.")

if __name__ == '__main__':
    main()
