from itertools import product, combinations
from random import Random

import pennylane as qml

from rl4pqc.architectures.helmi5q import HelmiArchitecture

import warnings

class PQGSampler:

    def __init__(self,
                 t_num,            # Trainable parameters - this is not needed and cause erros, this dependency shall be removed in future.
                 x_num,
                 w_num,
                 seed=None,
                 architecture=None,
                 x_chance=0.5,     # This was importand when chosing parameter for one argument gates but not the case for Helmi5Q
                 ) -> None:
        self.w_num = w_num 
        self.random_instance= Random(seed)
        self.t_symbols = [f"t_{i}" for i in range(t_num)]   #in fact index of trainable parameters does not matter so it can be nicer way to keep this dynamically like increasing counter with stash if something poped
        self.x_symbols = [f"x_{i}" for i in range(x_num)]
        self.t_queue = self.t_symbols[:]
        
    
        self.x_chance = x_chance

        self.available_symbols = self.t_symbols + self.x_symbols

        #Set up architecture
        self.architecture = architecture if architecture is not None else HelmiArchitecture()
        self.wires=self.architecture.get_specs()["wires"]
        #In case of using subset of qubits (add optional argument for specifying this subset?)
        if w_num < len(self.wires): 
            self.wires = self.wires[:w_num]
            self.architecture.set_active_wires(self.wires)
            
        self.native_ops=self.architecture.get_specs()["ops"]
        self.coupling_map=self.architecture.get_specs()["coupling_map"]

    def _next_t_symbol(self): 
        #Generate the next available trainable parameter name. ##Hotfix probably upgraded in future for use of circuits class
        if self.t_queue:
            return self.t_queue.pop(0)  # Get the next available t_*
        else:
            warnings.warn("All t_* parameters exhausted! Consider increasing t_num.")
            return None  # Or raise ValueError if you prefer strict handling
    
    def release_t_symbol(self, symbol):
        """Reinsert a released t_* parameter into the queue for reuse."""
        if symbol in self.t_symbols and symbol not in self.t_queue:
            self.t_queue.insert(0, symbol)   #Yes, not very optimal 

    def reset_symbols(self):
        self.t_queue = self.t_symbols[:]
        
        
    
    def _pqc_generator(self, action=None):
        
        while True:  # Infinite generator to yield gates dynamically
            gate_name = action.gate_name   #str   # Extract gate name from action 
            wires = action.wires            #tuple # Extract wires from action
            #action.get("param_type", None)                 #list  # Extract args types from action
            
            #Get gate
            if gate_name is None:
                gate_name = self.random_instance.choice(list(self.native_ops.keys()))
            if gate_name not in self.native_ops:
                raise ValueError(f"Unknown gate type: {gate_name}, chose one of {list(self.native_ops.keys())}")
            gate_info = self.native_ops[gate_name]
            
            #Sample wire(s)
            if wires is None:
                if gate_info['wires_num'] == 1:
                    wires = [self.random_instance.choice(self.wires)]
                elif gate_info['wires_num'] == 2:
                    wires = self.random_instance.choice(self.coupling_map)

            #You can specify parameters type in action but this make too much actions i think.
            #Sample parameter if needed
            if gate_info['params_num'] == 0:   # No params
                param = None
            elif gate_info['params_num'] == 1:  # Add one param of a type
                # implent doing something with param type
                param_type = self.random_instance.choices(
                    population=["x", "t"],
                    weights=[self.x_chance, 1 - self.x_chance], 
                    k=1
                )[0]
                if param_type == "x":
                    param = self.random_instance.choice(self.x_symbols)
                elif param_type == "t":
                    param = self._next_t_symbol()    
            elif gate_info['params_num'] == 2:  # Use first as x and t as second , maybe use x_chance here for general circuits?
                #again param type can be implemented to be also fetched from action
                param1 = self.random_instance.choice(self.x_symbols)
                param2 = self._next_t_symbol()
                param = [param1, param2]
            else:
                raise ValueError(f"Invalid number of parameters for gate {gate_name}")

            #Yield the constructed gate
            yield {
                "constructor": gate_info['fun'],
                "args": param,
                "wires": wires
            }

    
    def get_random_pqg(self, action :dict = None):
        """
        Fetch a random quantum gate using the generator.
        """
        return next(self._pqc_generator(action))

   

class RandomPQCGenerator:
    def __init__(self, w_num: int, x_num: int, t_num: int, seed=None, architecture=None, x_chance=0.5, max_gates=None) -> None:
        
        self.seed=seed
        self.random_instance= Random(self.seed)
        self.pqg_sampler = PQGSampler(t_num=t_num, x_num=x_num, w_num=w_num, seed=self.seed, architecture=architecture)
        self.PQC = list()
        self.max_gates = max_gates
        if self.max_gates is None:
            warnings.warn(f"No upperbound for number of gates this may cause erros!")
        
        self.actions = self.pqg_sampler.architecture.generate_action_set()
        
    #FUNCTIONAL
    @property
    def num_actions(self):
        """Returns the number of actions defined in the actions list."""
        return len(self.actions)
    
    @property
    def gates_num(self):
        """Returns the number of gates in the PQC."""
        return len(self.PQC)

    def reset(self):
        self.PQC = list()
        self.pqg_sampler.reset_symbols()
        
    def invoke_action(self, index):
        if index in range(len(self.actions)):
            action = self.actions[index]
            self.PQC.append(self.pqg_sampler.get_random_pqg(action))
            
            #if self.max_gates is not None: #Remove random gate if a maximal numer of gates is reached
             #   if len(self.PQC) > self.max_gates:     
              #      self.PQC.pop(self.random_instance.randint(0, len(self.PQC)-1))  # remove random gate
               #     warnings.warn(f"Max gates reached, removing random gate, this may not be desired behaviour.")
        else:
            raise ValueError(f"Invalid action index: {index}")
        
    """
    #ACTIONS ----------------## HARDCODED  SOLUTION but probably can just fetch gates name from the architecture class
    def add_prx(self):
        self.PQC.append(self.pqg_sampler.get_random_pqg('phased_rx_gate'))
        
    def add_cz(self):
        self.PQC.append(self.pqg_sampler.get_random_pqg('CZ'))
    """
        
    #EXPORTING, IMPORTING, CONVERSION state
    def to_qc(self, x, t, parameter_type='default'):
        """type: 'default' for preserving tensors with gradient tracking, 'numeric' for converting to numpy numbers for PyZX compatibility"""
        
        #no longer PyZX in the project, can it be safely removed?
        phi_values = {}
        for i, x_val in enumerate(x):
            if parameter_type == 'numeric':
                phi_values[f"x_{i}"] = x_val.numpy() if hasattr(x_val, 'numpy') else x_val
            else:
                phi_values[f"x_{i}"] = x_val

        for i, t_val in enumerate(t):
            if parameter_type == 'numeric':
                phi_values[f"t_{i}"] = t_val.numpy() if hasattr(t_val, 'numpy') else t_val
            else:
                phi_values[f"t_{i}"] = t_val


        qc = []
        trainable_params = []

        for i, pqg in enumerate(self.PQC):
            gate_copy = pqg.copy()
            phi_symbols = gate_copy.get("args")
            qc.append(gate_copy)
            if phi_symbols is not None:
                if len(phi_symbols) == 1:
                    qc[-1]["args"] = [phi_values[phi_symbol]]
                elif len(phi_symbols) == 2:
                    if phi_symbols[0] is None or phi_symbols[1] is None:
                        raise ValueError(f"symbol fetch was none, check config if 't_num' has big enough for circuit")
                    qc[-1]["args"] = [phi_values[phi_symbols[0]], phi_values[phi_symbols[1]]]
                else:
                    raise ValueError(f"Invalid number of parameters for gate {gate_name}")
                for sym in phi_symbols: 
                    if "t" in sym:
                       trainable_params.append(i)
        return qc, trainable_params
    
    def to_list(self):
        state = []
        for pqg in self.PQC:
            gate_type = pqg["constructor"].__name__
            args = pqg["args"]
            wires = pqg["wires"]
            state.append([gate_type, args, wires])
        return state
    
    def load_circuit(self, filepath: str = None, circuit: list = None):
        """
        Load a saved circuit into self.PQC, either from:
          - `circuit`, a list of [gate_type, args, wires], or
          - `filepath`, pointing to a JSON file with that same format.
        If both are provided, `circuit` takes precedence.
        """
        #Loading from file not tested.
        if circuit is None:
            if filepath is None:
                raise ValueError("Must provide either `circuit` or `filepath`")
            with open(filepath, 'r') as f:
                circuit = json.load(f)

        # Reset if anything exist 
        self.reset()

        for gate_type, args, wires in circuit:
            constructor = self.pqg_sampler.native_ops[gate_type]['fun']
            self.PQC.append({"constructor": constructor,
                            "args": args,
                            "wires": wires})

            # Remove any t_* symbols from the sampler queue
            if isinstance(args, list):
                for sym in args:
                    if isinstance(sym, str) and sym.startswith("t_"):
                        try:
                            self.pqg_sampler.t_queue.remove(sym)
                        except ValueError:
                            warnings.warn(f"Symbol {sym} was not in queue")

        return self