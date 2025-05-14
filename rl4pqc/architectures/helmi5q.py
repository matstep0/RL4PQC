from rl4pqc.architectures.architecture_base import QuantumArchitecture
import pennylane as qml

class HelmiArchitecture(QuantumArchitecture):
    # Custom gate
    def phased_rx_gate(self, theta, phi, wires):
        """Implement the Phased-RX gate using PennyLane's built-in gates."""
        qml.RZ(-phi, wires=wires)
        qml.RX(theta, wires=wires)
        qml.RZ(phi, wires=wires)
        
    def __init__(self):
        #Define the architecture     
        ops = {
            self.phased_rx_gate.__name__ : {"fun": self.phased_rx_gate, "params_num": 2, "wires_num": 1},
            qml.CZ.__name__: {"fun": qml.CZ, "params_num": 0, "wires_num": 2},
        }
        # Hardcode wires, coupling map and RL gate_encoding.        
        wires = [0, 1, 2, 3, 4]
        coupling_map = [[0, 2], [2, 0], [1, 2], [2, 1], [2, 3], [3, 2], [2, 4], [4, 2]]
        gate_encoding= {self.phased_rx_gate.__name__: 0,
                        qml.CZ.__name__: 1          # For state representation fetched for ML. ## Problematic from theoretical view point
                        }
        super().__init__("Helmi", ops, wires, coupling_map, gate_encoding)
    
    