from tqdm import tqdm


#from torchviz import make_dot
import pennylane as qml
from pennylane import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
#from sklearn.model_selection import train_test_split


from rl4pqc.pqcgen import RandomPQCGenerator


# Debug
from icecream import ic
#ic.enable()
ic.disable()

qml_devs= [
    'default.qubit',
    'lightning.qubit',  
]


class PQCEvaluator():
    def __init__(self,
                n_wires, 
                input_size,      # Number of features
                params_num,      # Params number 
                generator=None,  # PQCgenerator project class object.  Use it to load custom circuit
                num_classes=None, 
                lr=None,
                seed=None,       
                init_params=None,
                device_name=None  #Quantum device name
                ):
        
        # parameters
        self.lr = lr
        #meta parameters
        self.params_num = params_num
        self.num_classes = num_classes
        
        # Define seed generator
        self.seed = seed if seed is not None else np.random.randint(0, 2137) 
        ic(np.random.seed(self.seed))  #Track randomness in debug
        def seed_generator(start_seed):
            current_seed = start_seed
            while True:
                yield current_seed
                current_seed += 1
        self.my_seed_gen = seed_generator(self.seed)

        # Initialize the quantum device
        ## Check if the device name is one of the default devices
        if device_name not in qml_devs:
            print(f"Warning: The device '{device_name}' is not in the list of default devices: {qml_devs}")
        self.dev = qml.device(device_name, wires=n_wires)

        self.params = torch.tensor(np.random.rand(
            params_num) * np.pi, requires_grad=True)
        
        if init_params is not None:
            # Make sure `init_params` has length == params_num
            if len(init_params) != params_num:
                raise ValueError("Length of `init_params` must be equal to `params_num`")
     
            self.params = torch.tensor(init_params, requires_grad=True, dtype=torch.float32)
        else:
            self.params = torch.tensor(np.random.rand(params_num) * np.pi, 
                                       requires_grad=True, 
                                       dtype=torch.float32)
        
        #print(self.params)
        # Initialize the generator
        self.generator = generator if generator is not None else RandomPQCGenerator(
            n_wires, input_size, params_num, seed=self.seed)
        # for _ in range(n_actions):
        # self.generator.add_action()

        # Create a PyTorch optimizer (consider diffrent optimazers or setting options)
        self.optimizer = torch.optim.SGD([self.params], lr=lr, )



    #CIRCUIT CREATION---------------------------------------------------------------------------
    def create_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def quantum_function(x, params):
            # Quantum function definition using the state of generator
            qc, _ = self.generator.to_qc(x, params)
            for qg_desc in qc:
                constructor = qg_desc["constructor"]
                wires = qg_desc["wires"]
                args = qg_desc["args"]
                
                #print(constructor, wires, args)

                if args is None:  
                    constructor(wires=wires)
                else:  
                    constructor(*args, wires=wires)   #Fetch implicitly because gradients

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_classes)]

        self.quantum_function = quantum_function
            
    def __cross_entropy_loss(self, y, predictions):
        # Select the log probability corresponding to each label and sum
        ic(y, predictions)
        return -torch.sum(torch.log(predictions[torch.arange(len(y)), y.long()])) / len(y)

    def softmax(self, x, dim):
        exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def cost(self, X, y):
        ic(X.requires_grad, y.requires_grad)

        predictions_list = [torch.stack(
            self.quantum_function(x, self.params), dim=0) for x in X]
        predictions = torch.stack(predictions_list, dim=0)  # 2D tensor
        # Applying softmax along dimension 1 (the class dimension)
        predictions = self.softmax(predictions, dim=1)
        # Adjust the loss function
        ic(self.params.requires_grad)
        loss = self.__cross_entropy_loss(y, predictions)
        # return torch.sum(predictions)
        return loss

    def train_step(self, X: Tensor, y: Tensor):
        self.optimizer.zero_grad()
        loss = self.cost(X, y)
        ic(loss.requires_grad)
        # dot = make_dot(loss)
        # This will save it as 'computation_graph.pdf' in your current directory
        # dot.render(filename='computation_graph')

        if loss.requires_grad:
            # print("Before backward pass: ", self.params.grad)
            loss.backward()
            # print("After backward pass: ", self.params.grad)
            ic("before opt", self.params)
            self.optimizer.step()
            ic("after opt", self.params)
        else:
            pass
            # print("Skipping backward pass: loss does not require grad.")
        ic(self.params.grad)
        return loss.item()

    def predict_probabilities(self, X):
        # softmax return array
        # prediction_list = [torch.softmax(torch.tensor(self.quantum_function(x, self.params)), dim=0) for x in X]
        ic(self.params.grad)
        prediction_list = [torch.softmax(torch.tensor(
            self.quantum_function(x, self.params)), dim=0) for x in X]
        predictions = torch.stack(prediction_list, dim=0)
        return predictions

    def predict(self, X):
        with torch.no_grad():
            # prediction_list = [(torch.tensor(self.quantum_function(x, self.params)), dim=0) for x in X]
            predictions = self.predict_probabilities(X)
            predictions = torch.argmax(predictions, dim=1)
        return predictions

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute accuracy on the given dataset.
        """
        with torch.no_grad():
            acc = torch.sum(self.predict(X) == y.long()) / len(y)
        return acc.item()

    #Consider non obligatory validation_data argument
    def fit(self, train_data: tuple, valid_data: tuple, epochs=None, verbose=True):
        """
        Fit model on a traing data evaluating each time on validation data. 
        Passing numpy arrays and tensors should work here.
        Setting epochs to None skip training and makes only evaluation.
        
        Returns:
            train_losses: list of training losses
            val_losses: list of validation losses
            eval_train: list of training accuracies
            eval_valid: list of validation accuracies
            best_params: list of best parameters
        """
        g = torch.Generator()
        g.manual_seed(next(self.my_seed_gen))
        
        X_train, y_train = train_data
        X_valid, y_valid = valid_data
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        train_dataloader = DataLoader(train_dataset,
                                      shuffle=True,
                                      generator=g)
        valid_dataloader = DataLoader(valid_dataset)

        
        train_losses = []
        val_losses = []
        eval_train = []
        eval_valid = []
        best_accuracy = 0
        best_params = None
        
        # Quick-exit if no training requested
        if not epochs or epochs <= 0:
            if verbose:
                print("Skipping training (epochs <= 0); returning a single evaluation.")
            with torch.no_grad():
                train_acc = self.evaluate(X_train_tensor, y_train_tensor)
                valid_acc  = self.evaluate(X_valid_tensor, y_valid_tensor) if valid_data is not None else None
            return [], [], [train_acc], [valid_acc], self.params.clone().detach().tolist()
        
        #QML training
        for epoch in tqdm(range(epochs), desc="QML Fitting") if verbose else range(epochs):
            # Training
            train_loss = 0

            #consider using single batch at time
            for batch in train_dataloader:
                loss = self.train_step(batch[0], batch[1])
                train_loss += loss
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            # Validation  ##Maybe cross-validation here?
            val_loss = 0
            with torch.no_grad():
                for batch in valid_dataloader:
                    loss = self.cost(batch[0], batch[1])
                    val_loss += loss.item()
            val_loss /= len(valid_dataloader)
            val_losses.append(val_loss)

            # Prediction
            predicted_train = torch.sum(self.predict(
                X_train) == y_train_tensor) / len(y_train)
            predicted_valid = torch.sum(self.predict(
                X_valid) == y_valid_tensor) / len(y_valid)

            if predicted_valid > best_accuracy:
                best_accuracy = predicted_valid
                best_params = self.params.clone().detach().tolist()

            eval_train.append(predicted_train)
            eval_valid.append(predicted_valid)

            if verbose:
                print(
                    f'Epoch {epoch + 1}: '
                    f'Training Loss: {train_loss}, '
                    f'Validation Loss: {val_loss}, '
                    f'Prediction rate(train/valid): {predicted_train} / {predicted_valid}')

        return train_losses, val_losses, eval_train, eval_valid, best_params
