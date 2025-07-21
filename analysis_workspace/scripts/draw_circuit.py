import matplotlib.pyplot as plt
import json
import argparse
import matplotlib.patches as patches
import numpy as np

# Created for custom single use, can serve as a template
DATASET_CLASSES = {
    "iris": 3,
    "wine": 3,
    "breast_cancer": 2
}

def draw_quantum_circuit(circuit, qubit_labels=None, n_classes=None, figsize=(10, 2)):
    """
    Draws a quantum circuit from a list of gates.
    Each gate is a list: [gate_name, params, qubits]
    """
    n_qubits = len(qubit_labels) if qubit_labels is not None else 5
    #if qubit_labels is None:
        #qubit_labels = [f'Q{i+1}' for i in range(n_qubits)]
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(n_qubits)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(qubit_labels)
    ax.set_xticks([])

    # Draw horizontal lines for qubits
    for y in y_pos:
        ax.plot([-1, len(circuit)], [y, y], color='gray', lw=1)

    # Draw gates
    for i, gate in enumerate(circuit):
        name, params, targets = gate
        if name == "CZ":
            # Draw a vertical line between two qubits
            y1, y2 = targets
            ax.plot([i, i], [y1, y2], color='blue', lw=2)
            ax.scatter([i, i], [y1, y2], color='blue', s=80)
            ax.text(i, (max(y1,y2)+0.3), "CZ", color='blue', ha='center', va='center', fontsize=10, fontweight='bold')
        elif name == "phased_rx_gate":
            # Single qubit gate
            y = targets[0]
            ax.scatter(i, y, color='red', s=80)
            label = "PRX"
            if params:
                label += "\n" + ", ".join(params)
            ax.text(i, y+0.2, label, color='red', ha='center', va='bottom', fontsize=9)

    # Draw measurement blocks for each class
    if n_classes is not None:
        x = len(circuit)
        for y in range(n_classes):
            draw_measurement_box(ax, x, y)

    ax.set_xlim(-1, len(circuit)+2)
    ax.set_ylim(-1, n_qubits)
    #ax.set_title("Quantum Circuit Visualization")
    plt.tight_layout()

    # Remove the frame/border around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()


def draw_measurement_box(ax, x, y, box_width=0.5, box_height=0.5):
    # Draw the outer box
    rect = patches.Rectangle((x, y-0.25), box_width, box_height, linewidth=1.5, edgecolor='black', facecolor='white', alpha=0.5, zorder=5)
    ax.add_patch(rect)
    # Draw the semicircle (meter)
    semicircle = patches.Arc((x+box_width/2, y), box_width*0.7, box_height*0.7, theta1=0, theta2=180, color='black', lw=2, zorder=6)
    ax.add_patch(semicircle)
    # Draw the arrow (needle)
    angle = np.deg2rad(45)
    r = box_width*0.25
    ax.plot([x+box_width/2, x+box_width/2 + r*np.cos(angle)],
            [y, y + r*np.sin(angle)], color='black', lw=2, zorder=7)
    # Optionally, add a label
    #ax.text(x+box_width/2, y-0.05, 'M', color='black', ha='center', va='top', fontsize=14, fontweight='bold', zorder=8)


def main():
    parser = argparse.ArgumentParser(description="Draw quantum circuit from JSON file")
    parser.add_argument("json_path", help="Path to JSON file with circuit info")
    parser.add_argument("--line_number", type=int, default=None, help="Line number of circuit in JSON (matches 'line_number' field)")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)
    # Find the circuit with the matching line_number
    obj = None
    if args.line_number is not None:
        for entry in data:
            if entry.get("line_number", None) == args.line_number:
                obj = entry
                break
        if obj is None:
            print(f"No circuit found with line_number={args.line_number}")
            return
    # Or single circuit in JSON (not tested)
    #else:
        #obj = data[0] if isinstance(data, list) else data

    circuit = obj["circuit"]
    dataset = obj.get("dataset", None)
    n_classes = DATASET_CLASSES.get(dataset, None)

    # Infer number of qubits from circuit
    qubits = set()
    for gate in circuit:
        if len(gate) > 2 and gate[2]:
            qubits.update(gate[2])
    n_qubits = max(qubits)+1 if qubits else 5

    draw_quantum_circuit(
        circuit=circuit,
        qubit_labels=[f'Q{i+1}' for i in range(n_qubits)],
        n_classes=n_classes,
        figsize=(12, 3)
    )

if __name__ == "__main__":
    main()
