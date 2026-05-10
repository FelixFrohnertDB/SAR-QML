import pennylane as qml
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
from catalyst import qjit, value_and_grad
from tqdm import tqdm
from sklearn.model_selection import KFold


def _build_compiled_ad_step_double(
    qnode,
    opt,
    batch_size: int,
    epsilon: float,
    weight_decay: float,
    use_linear_readout: bool,
    quantum_param_size: int,
    readout_dim: int,
):
    def compiled_loss(weights, batch_x, batch_f, batch_y, batch_mask):
        total_loss = jnp.asarray(0.0, dtype=weights.dtype)

        if use_linear_readout:
            quantum_weights = weights[:quantum_param_size]
            readout_w = weights[quantum_param_size : quantum_param_size + readout_dim]
            readout_b = weights[quantum_param_size + readout_dim]

        for i in range(batch_size):
            if use_linear_readout:
                logit = qnode(batch_x[i], batch_f[i], quantum_weights, readout_w) + readout_b
                prob = jax.nn.sigmoid(logit)
            else:
                prob = (qnode(batch_x[i], batch_f[i], weights) + 1.0) / 2.0
            prob = jnp.clip(prob, epsilon, 1.0 - epsilon)
            sample_loss = -(batch_y[i] * jnp.log(prob) + (1.0 - batch_y[i]) * jnp.log(1.0 - prob))
            total_loss = total_loss + batch_mask[i] * sample_loss

        denom = jnp.maximum(jnp.sum(batch_mask), jnp.asarray(1.0, dtype=weights.dtype))
        bce = total_loss / denom
        if weight_decay > 0.0:
            bce = bce + weight_decay * jnp.sum(weights ** 2)
        return bce

    @qjit
    def run_step(weights, opt_state, batch_x, batch_f, batch_y, batch_mask):
        loss_val, grads = value_and_grad(compiled_loss)(weights, batch_x, batch_f, batch_y, batch_mask)
        updates, opt_state = opt.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, loss_val

    return run_step


def _build_compiled_ad_step_single(
    qnode,
    opt,
    batch_size: int,
    epsilon: float,
    weight_decay: float,
    use_linear_readout: bool,
    quantum_param_size: int,
    readout_dim: int,
    single_quantum_shape: Tuple[int, int],
):
    def compiled_loss(weights, batch_x, batch_y, batch_mask):
        total_loss = jnp.asarray(0.0, dtype=weights.dtype)

        if use_linear_readout:
            quantum_weights_flat = weights[:quantum_param_size]
            quantum_weights = quantum_weights_flat.reshape(single_quantum_shape)
            readout_w = weights[quantum_param_size : quantum_param_size + readout_dim]
            readout_b = weights[quantum_param_size + readout_dim]

        for i in range(batch_size):
            if use_linear_readout:
                logit = qnode(batch_x[i], quantum_weights, readout_w) + readout_b
                prob = jax.nn.sigmoid(logit)
            else:
                prob = (qnode(batch_x[i], weights) + 1.0) / 2.0
            prob = jnp.clip(prob, epsilon, 1.0 - epsilon)
            sample_loss = -(batch_y[i] * jnp.log(prob) + (1.0 - batch_y[i]) * jnp.log(1.0 - prob))
            total_loss = total_loss + batch_mask[i] * sample_loss

        denom = jnp.maximum(jnp.sum(batch_mask), jnp.asarray(1.0, dtype=weights.dtype))
        bce = total_loss / denom
        if weight_decay > 0.0:
            bce = bce + weight_decay * jnp.sum(weights ** 2)
        return bce

    @qjit
    def run_step(weights, opt_state, batch_x, batch_y, batch_mask):
        loss_val, grads = value_and_grad(compiled_loss)(weights, batch_x, batch_y, batch_mask)
        updates, opt_state = opt.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, loss_val

    return run_step


@dataclass
class PQCConfig:
    # architecture
    n_qubits: int = 8
    n_layers: int = 3
    n_x_layers: Optional[int] = None
    n_f_layers: Optional[int] = None

    # training
    n_epochs: int = 200
    learning_rate: float = 0.005  # Increased slightly for Adjoint
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
    weight_init_scale: float = 0.03
    weight_decay: float = 0.0
    epsilon: float = 1e-7
    log_interval: int = 10
    eval_every: int = 5
    show_progress: bool = False
    catalyst_batch_size: int = 128
    readout_mode: str = "single"
    readout_wires: Optional[int] = None


class PQCClassifier:
    def __init__(self, config: PQCConfig, use_double_reuploading: bool = False):
        self.config = config
        self.use_double_reuploading = use_double_reuploading
        self.readout_mode = self.config.readout_mode.lower()
        if self.readout_mode not in {"single", "linear"}:
            raise ValueError("readout_mode must be 'single' or 'linear'")
        self.use_linear_readout = self.readout_mode == "linear"
        self.x_layers = self.config.n_x_layers if self.config.n_x_layers is not None else self.config.n_layers
        self.f_layers = self.config.n_f_layers if self.config.n_f_layers is not None else self.config.n_layers
        self.max_double_layers = max(self.x_layers, self.f_layers)
        self.readout_dim = self._get_readout_dim()

        # Device: "adjoint" is preferred for lightning.qubit
        self.dev = qml.device("lightning.qubit", wires=self.config.n_qubits)

        # We allow batching by not specifying input signature constraints strictly
        self.qnode_single = qml.QNode(self._circuit_single, self.dev, interface="jax", diff_method="adjoint")
        self.qnode_double = qml.QNode(self._circuit_double, self.dev, interface="jax", diff_method="adjoint")

        self.training_history: Dict[str, Any] = {}

    def _get_readout_dim(self) -> int:
        if not self.use_linear_readout:
            return 1
        requested = self.config.readout_wires if self.config.readout_wires is not None else self.config.n_qubits
        return max(1, min(int(requested), self.config.n_qubits))

    def _quantum_param_size(self) -> int:
        if self.use_double_reuploading:
            return (self.x_layers + self.f_layers) * self.config.n_qubits
        return self.x_layers * self.config.n_qubits

    def _split_joint_weights(self, weights):
        if not self.use_linear_readout:
            return weights, None, None

        q_size = self._quantum_param_size()
        q_weights = weights[:q_size]
        readout_w = weights[q_size : q_size + self.readout_dim]
        readout_b = weights[q_size + self.readout_dim]

        if not self.use_double_reuploading:
            q_weights = q_weights.reshape(self.x_layers, self.config.n_qubits)

        return q_weights, readout_w, readout_b

    def _linear_readout_observable(self, readout_w):
        observables = [qml.PauliZ(i) for i in range(self.readout_dim)]
        return qml.Hamiltonian(readout_w, observables)

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
    def _circuit_single(self, inputs, weights, readout_w=None):
        """
        inputs: shape (batch_size, n_qubits)
        weights: shape (n_layers, n_qubits)
        """
        for l in range(self.x_layers):
            self.angle_encoding(inputs)
            self.variational_layer(weights[l])
        if self.use_linear_readout:
            if readout_w is not None:
                return qml.expval(self._linear_readout_observable(readout_w))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.readout_dim)]
        return qml.expval(qml.PauliZ(0))

    def _circuit_double(self, inputs_x, inputs_f, weights, readout_w=None):
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
        if self.use_linear_readout:
            if readout_w is not None:
                return qml.expval(self._linear_readout_observable(readout_w))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.readout_dim)]
        return qml.expval(qml.PauliZ(0))

    # ------------------------------
    # Vectorized Prediction
    # ------------------------------
    def predict(self, X, weights):
        """
        Predicts for the whole batch X at once using broadcasting.
        """
        if self.use_linear_readout:
            q_weights, readout_w, readout_b = self._split_joint_weights(weights)
            if self.use_double_reuploading:
                X_data, F_data = X
                res = self.qnode_double(X_data, F_data, q_weights)
            else:
                res = self.qnode_single(X, q_weights)

            features = jnp.asarray(res)
            if features.ndim == 1:
                # Single-sample or already reduced path.
                logits = jnp.dot(readout_w, features) + readout_b
            elif features.ndim == 2:
                # PennyLane may return (readout_dim, batch) for multi-measurement outputs.
                if features.shape[0] == self.readout_dim:
                    feature_matrix = jnp.swapaxes(features, 0, 1)
                elif features.shape[1] == self.readout_dim:
                    feature_matrix = features
                else:
                    raise ValueError(
                        f"Unexpected linear readout feature shape {features.shape} for dim={self.readout_dim}"
                    )
                logits = feature_matrix @ readout_w + readout_b
            else:
                raise ValueError(f"Unexpected linear readout feature rank: {features.ndim}")

            return jax.nn.sigmoid(logits)

        if self.use_double_reuploading:
            X_data, F_data = X
            # Pass full batch directly
            res = self.qnode_double(X_data, F_data, weights)
        else:
            # Pass full batch directly
            res = self.qnode_single(X, weights)

        # Ensure result is a proper array
        res = jnp.asarray(res)

        # Map expectation [-1, 1] to probability [0, 1]
        return (res + 1) / 2

    def loss(self, weights, X, y):
        """Vectorized Binary Cross Entropy with optional L2 weight decay."""
        probs = self.predict(X, weights)
        
        # Numerical stability clip
        probs = jnp.clip(probs, self.config.epsilon, 1 - self.config.epsilon)
        
        bce = -jnp.mean(y * jnp.log(probs) + (1 - y) * jnp.log(1 - probs))

        if self.config.weight_decay > 0.0:
            l2 = jnp.sum(weights ** 2)
            return bce + self.config.weight_decay * l2
        return bce

    def accuracy(self, weights, X, y):
        probs = self.predict(X, weights)
        preds = (probs > 0.5).astype(int)
        # y may be soft labels (float in [0,1]); threshold at 0.5 to get hard targets.
        y_hard = (jnp.asarray(y) > 0.5).astype(int)
        return float(jnp.mean(preds == y_hard))

    # ------------------------------
    # Helpers
    # ------------------------------
    def init_weights(self):
        key = jax.random.PRNGKey(self.config.random_state)
        return self._init_weights_for_key(key)

    def _init_weights_for_key(self, key):
        if self.use_linear_readout:
            n_params = self._quantum_param_size() + self.readout_dim + 1
            return self.config.weight_init_scale * jax.random.normal(key, (n_params,))

        if self.use_double_reuploading:
            # Flat 1D: [x_layers * n_qubits] x-weights then [f_layers * n_qubits] f-weights.
            n_params = (self.x_layers + self.f_layers) * self.config.n_qubits
            return self.config.weight_init_scale * jax.random.normal(key, (n_params,))
        else:
            shape = (self.x_layers, self.config.n_qubits)
            return self.config.weight_init_scale * jax.random.normal(key, shape)

    def get_optimizer(self):
        if self.config.optimizer.lower() == "adam":
            return optax.adam(
                self.config.learning_rate,
                b1=self.config.adam_beta1,
                b2=self.config.adam_beta2,
                eps=self.config.adam_eps,
            )

        return optax.sgd(self.config.learning_rate)

    def _as_jax_data(self, data):
        if self.use_double_reuploading:
            x_data, f_data = data
            return jnp.asarray(x_data), jnp.asarray(f_data)
        return jnp.asarray(data)

    def _make_compiled_ad_step(self, batch_size: int):
        if batch_size <= 0:
            return None

        opt = self.get_optimizer()
        epsilon = float(self.config.epsilon)
        weight_decay = float(self.config.weight_decay)

        if self.use_double_reuploading:
            return _build_compiled_ad_step_double(
                self.qnode_double,
                opt,
                batch_size,
                epsilon,
                weight_decay,
                self.use_linear_readout,
                self._quantum_param_size(),
                self.readout_dim,
            )

        return _build_compiled_ad_step_single(
            self.qnode_single,
            opt,
            batch_size,
            epsilon,
            weight_decay,
            self.use_linear_readout,
            self._quantum_param_size(),
            self.readout_dim,
            (self.x_layers, self.config.n_qubits),
        )

    def _make_jax_fullbatch_step(self):
        """Build a jax.jit-compiled full-batch training step.

        Avoids Catalyst's O(N) MLIR unrolling — safe for N up to ~10k.
        Uses PennyLane's native batch execution + adjoint VJP registered
        via jax.custom_vjp, so gradients are exact (adjoint method).
        """
        opt = self.get_optimizer()
        loss_fn = self.loss  # captures self

        if self.use_double_reuploading:
            @jax.jit
            def run_step(weights, opt_state, train_x, train_f, train_y):
                loss_val, grads = jax.value_and_grad(loss_fn)(weights, (train_x, train_f), train_y)
                updates, new_opt_state = opt.update(grads, opt_state, weights)
                new_weights = optax.apply_updates(weights, updates)
                return new_weights, new_opt_state, loss_val
        else:
            @jax.jit
            def run_step(weights, opt_state, train_x, train_y):
                loss_val, grads = jax.value_and_grad(loss_fn)(weights, train_x, train_y)
                updates, new_opt_state = opt.update(grads, opt_state, weights)
                new_weights = optax.apply_updates(weights, updates)
                return new_weights, new_opt_state, loss_val

        return opt, run_step

    def _make_jax_minibatch_step(self, batch_size: int):
        """Build a jax.jit-compiled minibatch step with mask support.

        Used for linear readout mode to avoid Catalyst lowering failures while
        keeping minibatch training behavior.
        """
        opt = self.get_optimizer()
        epsilon = float(self.config.epsilon)
        weight_decay = float(self.config.weight_decay)

        if self.use_double_reuploading:
            @jax.jit
            def run_step(weights, opt_state, batch_x, batch_f, batch_y, batch_mask):
                probs = self.predict((batch_x, batch_f), weights)
                probs = jnp.clip(probs, epsilon, 1.0 - epsilon)
                sample_losses = -(batch_y * jnp.log(probs) + (1.0 - batch_y) * jnp.log(1.0 - probs))
                denom = jnp.maximum(jnp.sum(batch_mask), jnp.asarray(1.0, dtype=weights.dtype))
                bce = jnp.sum(batch_mask * sample_losses) / denom
                if weight_decay > 0.0:
                    bce = bce + weight_decay * jnp.sum(weights ** 2)

                grads = jax.grad(lambda w: (
                    jnp.sum(
                        batch_mask
                        * (
                            -(
                                batch_y * jnp.log(jnp.clip(self.predict((batch_x, batch_f), w), epsilon, 1.0 - epsilon))
                                + (1.0 - batch_y)
                                * jnp.log(1.0 - jnp.clip(self.predict((batch_x, batch_f), w), epsilon, 1.0 - epsilon))
                            )
                        )
                    )
                    / jnp.maximum(jnp.sum(batch_mask), jnp.asarray(1.0, dtype=w.dtype))
                    + (weight_decay * jnp.sum(w ** 2) if weight_decay > 0.0 else 0.0)
                ))(weights)

                updates, new_opt_state = opt.update(grads, opt_state, weights)
                new_weights = optax.apply_updates(weights, updates)
                return new_weights, new_opt_state, bce
        else:
            @jax.jit
            def run_step(weights, opt_state, batch_x, batch_y, batch_mask):
                probs = self.predict(batch_x, weights)
                probs = jnp.clip(probs, epsilon, 1.0 - epsilon)
                sample_losses = -(batch_y * jnp.log(probs) + (1.0 - batch_y) * jnp.log(1.0 - probs))
                denom = jnp.maximum(jnp.sum(batch_mask), jnp.asarray(1.0, dtype=weights.dtype))
                bce = jnp.sum(batch_mask * sample_losses) / denom
                if weight_decay > 0.0:
                    bce = bce + weight_decay * jnp.sum(weights ** 2)

                grads = jax.grad(lambda w: (
                    jnp.sum(
                        batch_mask
                        * (
                            -(
                                batch_y * jnp.log(jnp.clip(self.predict(batch_x, w), epsilon, 1.0 - epsilon))
                                + (1.0 - batch_y)
                                * jnp.log(1.0 - jnp.clip(self.predict(batch_x, w), epsilon, 1.0 - epsilon))
                            )
                        )
                    )
                    / jnp.maximum(jnp.sum(batch_mask), jnp.asarray(1.0, dtype=w.dtype))
                    + (weight_decay * jnp.sum(w ** 2) if weight_decay > 0.0 else 0.0)
                ))(weights)

                updates, new_opt_state = opt.update(grads, opt_state, weights)
                new_weights = optax.apply_updates(weights, updates)
                return new_weights, new_opt_state, bce

        return opt, run_step

    def _run_eager_chunk(self, weights, opt, opt_state, train_data, y_train, steps: int):
        losses = []
        for _ in range(steps):
            loss_val, grads = jax.value_and_grad(self.loss)(weights, train_data, y_train)
            updates, opt_state = opt.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)
            losses.append(float(loss_val))
        return weights, opt_state, losses

    # ------------------------------
    # Training
    # ------------------------------
    def train_fold(self, train_data, y_train, val_data, y_val, fold_num):
        # Deterministic seed per fold
        fold_seed = self.config.random_state + fold_num
        key = jax.random.PRNGKey(fold_seed)
        weights = self._init_weights_for_key(key)

        train_data = self._as_jax_data(train_data)
        val_data = self._as_jax_data(val_data)
        y_train = jnp.asarray(y_train)
        y_val = jnp.asarray(y_val)

        best_val_acc = -1.0
        best_weights = weights
        patience_counter = 0
        
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        eval_every = max(1, int(self.config.eval_every))

        # Training mode selection:
        #   catalyst_batch_size > 0 → prefer Catalyst qjit minibatch AD
        #   catalyst_batch_size = 0 → JAX jit full-batch AD
        # For scalar-observable linear readout we first try the Catalyst path,
        # and only fall back to JAX minibatch if compilation still fails.
        n_train = int(train_data[0].shape[0] if self.use_double_reuploading else train_data.shape[0])
        use_fullbatch_jax = (self.config.catalyst_batch_size == 0)

        if use_fullbatch_jax:
            opt, step_fn = self._make_jax_fullbatch_step()
        else:
            opt = self.get_optimizer()
            batch_size = max(1, min(int(self.config.catalyst_batch_size), n_train))
            step_fn = self._make_compiled_ad_step(batch_size)
        opt_state = opt.init(weights)
        linear_fallback_active = False

        chunk_iterator = range(0, self.config.n_epochs, eval_every)
        if self.config.show_progress:
            chunk_iterator = tqdm(chunk_iterator, desc=f"Fold {fold_num}", leave=False)

        for chunk_start in chunk_iterator:
            remaining = self.config.n_epochs - chunk_start
            steps_this_chunk = min(eval_every, remaining)

            loss_values = []
            for _ in range(steps_this_chunk):
                if use_fullbatch_jax:
                    # Full-batch: one step over the entire training set.
                    if self.use_double_reuploading:
                        train_x, train_f = train_data
                        weights, opt_state, batch_loss = step_fn(
                            weights, opt_state, train_x, train_f, y_train
                        )
                    else:
                        weights, opt_state, batch_loss = step_fn(
                            weights, opt_state, train_data, y_train
                        )
                    loss_values.append(float(batch_loss))
                else:
                    weighted_loss_sum = 0.0
                    seen = 0

                    # Shuffle once per optimization step (outside qjit) while
                    # keeping compiled step input shapes unchanged.
                    key, perm_key = jax.random.split(key)
                    perm = jax.random.permutation(perm_key, n_train)

                    if self.use_double_reuploading:
                        train_x, train_f = train_data
                        train_x_epoch = train_x[perm]
                        train_f_epoch = train_f[perm]
                    else:
                        train_x = train_data
                        train_x_epoch = train_x[perm]
                    y_train_epoch = y_train[perm]

                    for b_start in range(0, n_train, batch_size):
                        b_end = min(n_train, b_start + batch_size)
                        bs = b_end - b_start
                        pad = batch_size - bs

                        batch_mask = jnp.concatenate(
                            [
                                jnp.ones((bs,), dtype=weights.dtype),
                                jnp.zeros((pad,), dtype=weights.dtype),
                            ]
                        )

                        if self.use_double_reuploading:
                            batch_x = train_x_epoch[b_start:b_end]
                            batch_f = train_f_epoch[b_start:b_end]
                            batch_y = y_train_epoch[b_start:b_end]
                            if pad > 0:
                                batch_x = jnp.pad(batch_x, ((0, pad), (0, 0)))
                                batch_f = jnp.pad(batch_f, ((0, pad), (0, 0)))
                                batch_y = jnp.pad(batch_y, ((0, pad),))
                            try:
                                weights, opt_state, batch_loss = step_fn(
                                    weights,
                                    opt_state,
                                    batch_x,
                                    batch_f,
                                    batch_y,
                                    batch_mask,
                                )
                            except Exception:
                                if (not self.use_linear_readout) or linear_fallback_active:
                                    raise
                                print("Linear readout Catalyst compile failed; falling back to JAX minibatch training.")
                                opt, step_fn = self._make_jax_minibatch_step(batch_size)
                                linear_fallback_active = True
                                weights, opt_state, batch_loss = step_fn(
                                    weights,
                                    opt_state,
                                    batch_x,
                                    batch_f,
                                    batch_y,
                                    batch_mask,
                                )
                        else:
                            batch_x = train_x_epoch[b_start:b_end]
                            batch_y = y_train_epoch[b_start:b_end]
                            if pad > 0:
                                batch_x = jnp.pad(batch_x, ((0, pad), (0, 0)))
                                batch_y = jnp.pad(batch_y, ((0, pad),))
                            try:
                                weights, opt_state, batch_loss = step_fn(
                                    weights,
                                    opt_state,
                                    batch_x,
                                    batch_y,
                                    batch_mask,
                                )
                            except Exception:
                                if (not self.use_linear_readout) or linear_fallback_active:
                                    raise
                                print("Linear readout Catalyst compile failed; falling back to JAX minibatch training.")
                                opt, step_fn = self._make_jax_minibatch_step(batch_size)
                                linear_fallback_active = True
                                weights, opt_state, batch_loss = step_fn(
                                    weights,
                                    opt_state,
                                    batch_x,
                                    batch_y,
                                    batch_mask,
                                )

                        weighted_loss_sum += float(batch_loss) * bs
                        seen += bs

                    loss_values.append(weighted_loss_sum / seen)

            history["train_loss"].extend(loss_values)

            if steps_this_chunk > 1:
                history["val_loss"].extend([float("nan")] * (steps_this_chunk - 1))
                history["train_acc"].extend([float("nan")] * (steps_this_chunk - 1))
                history["val_acc"].extend([float("nan")] * (steps_this_chunk - 1))

            train_acc = self.accuracy(weights, train_data, y_train)
            val_loss = self.loss(weights, val_data, y_val)
            val_acc = self.accuracy(weights, val_data, y_val)

            history["val_loss"].append(float(val_loss))
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            last_loss = loss_values[-1]
            if self.config.show_progress:
                chunk_iterator.set_postfix({"t_loss": f"{last_loss:.3f}", "v_acc": f"{val_acc:.3f}"})

            if val_acc > best_val_acc + self.config.min_improvement:
                best_val_acc = val_acc
                best_weights = jnp.array(weights)
                patience_counter = 0
            else:
                patience_counter += 1

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
        print(f"\nResult: {jnp.mean(jnp.asarray(accs)):.4f} ± {jnp.std(jnp.asarray(accs)):.4f}")
        return fold_results