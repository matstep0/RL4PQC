



class QuantumArchitecture:
    # Class-level attributes (shared across all instances)
    all_wires = []
    coupling_map = []
    
    from dataclasses import dataclass
    @dataclass(frozen=True, slots=False)
    class ActionSpec:
        gate_name: str             # e.g. "phased_rx_gate"
        wires: list[int, ...]     # ordered qubit indices
        #can easily extend for also chosing specific parameters (trainable vs features)

    def __init__(self, name, ops, wires, coupling_map, gate_representation=None):
        """
        Initialize the architecture.

        Args:
            name (str): Name of the architecture.
            ops (dict): Dictionary of quantum operations.
            wires (list): List of all qubits in the architecture.
            coupling_map (list): Full list of qubit connections.
        """
        self.name = name
        self.ops = ops

        # Set full specifications as class-level variables
        self.__class__.all_wires = wires
        self.__class__.coupling_map = coupling_map
        self.__class__.gate_representation = gate_representation

        # Initialize active specs to match full specs
        self.active_wires = list(wires)
        self.active_coupling_map = list(coupling_map)

    def set_active_wires(self, wires_subset):
        """
        Set the active subset of wires and update the active coupling map.

        Args:
            wires_subset (list): Subset of wires to activate.
        """
        if not all(wire in self.all_wires for wire in wires_subset):
            raise ValueError(f"Some qubits in {wires_subset} are not part of the architecture.")
        self.active_wires = wires_subset
        self._update_coupling_map()

    def _update_coupling_map(self):
        """
        Update the active coupling map to reflect the current active wires.
        """
        self.active_coupling_map = [
            pair for pair in self.coupling_map if all(q in self.active_wires for q in pair)
        ]

    def get_all_wires(self):
        """
        Get the full list of wires from the architecture.
        """
        return self.__class__.all_wires

    def get_specs(self):
        """
        Get the active specifications of the architecture.

        Returns:
            dict: Dictionary containing `ops`, `active_wires`, and `active_coupling_map`.
        """
        return {
            "ops": self.ops,
            "wires": self.active_wires,
            "coupling_map": self.active_coupling_map,
        }
        
    def generate_action_set(self) -> list["QuantumArchitecture.ActionSpec"]:
        """
        Enumerate all valid (gate , wire-tuple) combinations.
        Returns a list whose index is the RL action id.
        """
        actions: list[QuantumArchitecture.ActionSpec] = []

        for gname, ginfo in self.ops.items():              # native gates
            wn = ginfo["wires_num"]

            if wn == 1:                                    # one-qubit gates
                for w in self.active_wires:
                    actions.append(self.ActionSpec(gname, [w]))

            elif wn == 2:                                  # two-qubit gates
                for edge in self.active_coupling_map:      # directed list
                    #micro fuckup - some gates (like CZ are symetric and 2q action is duplicated.. )
                    # possibly modify actionspec by symetric=true or check.
                    actions.append(self.ActionSpec(gname, edge))

            else:                                          # >2-qubit not yet
                raise NotImplementedError

        return actions

    #I dont think this is needed
    def get_gate_encoding(self):
        """
        Get the gate encoding for the architecture.
        Returns:
            dict: Dictionary mapping gate names to their encodings.
        """
        if self.gate_representation is None:
            raise ValueError("Gate representation is not defined.")
        return self.gate_representation