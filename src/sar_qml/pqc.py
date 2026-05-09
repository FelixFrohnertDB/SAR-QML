import pennylane as qml
from pennylane import numpy as pp
from dataclasses import dataclass
from typing import Optional, Any, Dict
from tqdm import tqdm
from sklearn.model_selection import KFold


@dataclass
class PQCConfig:
    # architecture
    n_qubits: int = 8
    n_layers: int = 3
    n_x_layers: Optional[int] = None
    n_f_layers: Optional[int] = None

    # training
    n_epochs: int = 200
    learning_rate: float = 0.01  # Increased slightly for Adjoint
    optimizer: str = "adam" 

    # optimizer params
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # cross-validation
    n_folds: int = 5
    random_state: int = 42

    # early stopping
    patience: int = 15
    min_improvement: float = 1e-3

    # regularization
    weight_init_scale: float = 0.01
    weight_decay: float = 0.0
    epsilon: float = 1e-7
    log_interval: int = 10
    eval_every: int = 5
    show_progress: bool = False


class PQCClassifier:
    def __init__(self, config: PQCConfig, use_double_reuploading: bool = False):
        self.config = config
        self.use_double_reuploading = use_double_reuploading
        self.x_layers = self.config.n_x_layers if self.config.n_x_layers is not None else self.config.n_layers
        self.f_layers = self.config.n_f_layers if self.config.n_f_layers is not None else self.config.n_layers
        self.max_double_layers = max(self.x_layers, self.f_layers)

        # Device: "adjoint" is preferred for lightning.qubit
        self.dev = qml.device("lightning.qubit", wires=self.config.n_qubits)

        # We allow batching by not specifying input signature constraints strictly
        self.qnode_single = qml.QNode(self._circuit_single, self.dev, diff_method="adjoint")    
        self.qnode_double = qml.QNode(self._circuit_double, self.dev, diff_method="adjoint")

        self.training_history: Dict[str, Any] = {}

    # ------------------------------
    # Circuit components
    # ------------------------------
    def angle_encoding(self, x):
        """
        Encodes data. 
        If x is shape (batch_size, n_qubits), AngleEmbedding broadcasts automatically.
        """
        qml.AngleEmbedding(x, wires=range(self.config.n_qubits), rotation="Y")

    def variational_layer(self, w):
        """
        w is shape (n_qubits,). 
        If we are broadcasting data, w remains constant across the batch.
        """
        for i in range(self.config.n_qubits):
            qml.RY(w[i], wires=i)
        
        # Efficient Ring Entangler
        for i in range(self.config.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.config.n_qubits - 1, 0])

    # ------------------------------
    # Circuits (Vectorized)
    # ------------------------------
    def _circuit_single(self, inputs, weights):
        """
        inputs: shape (batch_size, n_qubits)
        weights: shape (n_layers, n_qubits)
        """
        for l in range(self.x_layers):
            self.angle_encoding(inputs)
            self.variational_layer(weights[l])
        return qml.expval(qml.PauliZ(0))

    def _circuit_double(self, inputs_x, inputs_f, weights):
        """
        inputs_x: shape (batch_size, n_qubits)
        inputs_f: shape (batch_size, n_qubits)
        weights: flat 1D array of length (x_layers + f_layers) * n_qubits.
            First x_layers * n_qubits elements are x-branch weights;
            remaining f_layers * n_qubits are f-branch weights.
        """
        nq = self.config.n_qubits
        wx = weights[: self.x_layers * nq].reshape(self.x_layers, nq)
        wf = weights[self.x_layers * nq :].reshape(self.f_layers, nq)
        for l in range(self.max_double_layers):
            if l < self.x_layers:
                self.angle_encoding(inputs_x)
                self.variational_layer(wx[l])
            if l < self.f_layers:
                self.angle_encoding(inputs_f)
                self.variational_layer(wf[l])
        return qml.expval(qml.PauliZ(0))

    # ------------------------------
    # Vectorized Prediction
    # ------------------------------
    def predict(self, X, weights):
        """
        Predicts for the whole batch X at once using broadcasting.
        """
        if self.use_double_reuploading:
            X_data, F_data = X
            # Pass full batch directly
            res = self.qnode_double(X_data, F_data, weights)
        else:
            # Pass full batch directly
            res = self.qnode_single(X, weights)
        
        # Ensure result is a proper array
        res = pp.array(res) 
        
        # Map expectation [-1, 1] to probability [0, 1]
        return (res + 1) / 2

    def loss(self, weights, X, y):
        """Vectorized Binary Cross Entropy with optional L2 weight decay."""
        probs = self.predict(X, weights)
        
        # Numerical stability clip
        probs = pp.clip(probs, self.config.epsilon, 1 - self.config.epsilon)
        
        bce = -pp.mean(y * pp.log(probs) + (1 - y) * pp.log(1 - probs))

        if self.config.weight_decay > 0.0:
            l2 = pp.sum(weights ** 2)
            return bce + self.config.weight_decay * l2
        return bce

    def accuracy(self, weights, X, y):
        probs = self.predict(X, weights)
        preds = (probs > 0.5).astype(int)
        # y may be soft labels (float in [0,1]); threshold at 0.5 to get hard targets.
        y_hard = (pp.array(y) > 0.5).astype(int)
        return float(pp.mean(preds == y_hard))

    # ------------------------------
    # Helpers
    # ------------------------------
    def init_weights(self):
        if self.use_double_reuploading:
            # Flat 1D: [x_layers * n_qubits] x-weights then [f_layers * n_qubits] f-weights.
            n_params = (self.x_layers + self.f_layers) * self.config.n_qubits
            return pp.array(
                self.config.weight_init_scale * pp.random.randn(n_params),
                requires_grad=True,
            )
        else:
            shape = (self.x_layers, self.config.n_qubits)
            return pp.array(
                self.config.weight_init_scale * pp.random.randn(*shape),
                requires_grad=True,
            )

    def get_optimizer(self):
        if self.config.optimizer.lower() == "adam":
            return qml.AdamOptimizer(
                stepsize=self.config.learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
                eps=self.config.adam_eps,
            )

        return qml.GradientDescentOptimizer(stepsize=self.config.learning_rate)

    # ------------------------------
    # Training
    # ------------------------------
    def train_fold(self, train_data, y_train, val_data, y_val, fold_num):
        # Deterministic seed per fold
        pp.random.seed(self.config.random_state + fold_num)

        weights = self.init_weights()
        opt = self.get_optimizer()

        best_val_acc = -1.0
        best_weights = weights
        patience_counter = 0
        
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Data unpacking logic handled inside loss/predict via 'train_data' tuple/array
        
        eval_every = max(1, int(self.config.eval_every))
        epoch_iterator = range(self.config.n_epochs)
        if self.config.show_progress:
            epoch_iterator = tqdm(epoch_iterator, desc=f"Fold {fold_num}", leave=False)

        for epoch in epoch_iterator:
            
            # Optimization step
            # Note: We pass the *entire* train_data batch here
            weights, loss_val = opt.step_and_cost(lambda w: self.loss(w, train_data, y_train), weights)

            history["train_loss"].append(float(loss_val))

            should_eval = ((epoch + 1) % eval_every == 0) or (epoch == self.config.n_epochs - 1)
            if should_eval:
                train_acc = self.accuracy(weights, train_data, y_train)
                val_loss = self.loss(weights, val_data, y_val)
                val_acc = self.accuracy(weights, val_data, y_val)

                history["val_loss"].append(float(val_loss))
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)

                if self.config.show_progress:
                    epoch_iterator.set_postfix({"t_loss": f"{loss_val:.3f}", "v_acc": f"{val_acc:.3f}"})

                # Early stopping is checked on evaluated epochs only.
                if val_acc > best_val_acc + self.config.min_improvement:
                    best_val_acc = val_acc
                    best_weights = pp.array(weights, requires_grad=False)
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                history["val_loss"].append(float("nan"))
                history["train_acc"].append(float("nan"))
                history["val_acc"].append(float("nan"))
            
            if patience_counter >= self.config.patience:
                break
        
        return best_weights, history, best_val_acc

    def train(self, data, y_data):
        print(f"Starting {self.config.n_folds}-Fold CV | Mode: {'Double' if self.use_double_reuploading else 'Single'}")
        
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state)
        
        # Handle unpacking for KFold splitting
        if self.use_double_reuploading:
            X_all, F_all = data
            splitter = X_all
        else:
            splitter = data

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(splitter), 1):
            y_train, y_val = y_data[train_idx], y_data[val_idx]

            if self.use_double_reuploading:
                train_data = (X_all[train_idx], F_all[train_idx])
                val_data = (X_all[val_idx], F_all[val_idx])
            else:
                train_data = data[train_idx]
                val_data = data[val_idx]

            weights, hist, final_acc = self.train_fold(train_data, y_train, val_data, y_val, fold)
            
            fold_results.append({
                "fold": fold,
                "final_acc": final_acc,
                "history": hist
            })
            
            print(f"Fold {fold} finished. Best Val Acc: {final_acc:.4f}")

        accs = [r['final_acc'] for r in fold_results]
        print(f"\nResult: {pp.mean(accs):.4f} ± {pp.std(accs):.4f}")
        return fold_results