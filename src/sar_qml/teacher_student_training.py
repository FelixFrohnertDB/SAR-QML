"""Training and evaluation of teacher-student classifiers.

A student predicts ``p(y=1|x) = sigmoid(beta * v(x, f; theta))`` and is
trained by binary cross-entropy: full-batch Adam from several prior samples
in parallel, followed by an L-BFGS polish of the best Adam results. Model
selection uses the training loss only.

Two fit functions exist because the two notebooks compile the L-BFGS
objective differently, which can change floating-point results in the last
bits; each cached result set was produced by one of them:

* :func:`fit_classifier_local_jit` compiles a fresh objective with the
  training data baked in for every fit (learning curves);
* :func:`fit_classifier_shared_jit` uses one compiled objective, from
  :func:`make_value_and_grad`, that takes the data as arguments (query scan).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .teacher_student import forward, rng_from, sample_free_prior


def bce_from_logits(logits, labels):
    """Mean binary cross-entropy computed from logits."""
    labels = jnp.asarray(labels)
    return jnp.mean(jax.nn.softplus(logits) - labels * logits)


def classification_loss(free_parameters, x, labels, f, cfg, beta):
    """BCE of the student logits beta * v."""
    score = forward(cfg.expand(free_parameters), x, f, cfg)
    return bce_from_logits(beta * score, labels)


def make_batched_adam(
    cfg,
    *,
    beta,
    n_steps,
    learning_rate,
    gradient_clip=10.0,
):
    """Jitted full-batch Adam with clipping and cosine schedule, vectorized over starts.

    Returns ``run(initial_parameters, x, labels, f) -> (parameters, losses)``
    where ``initial_parameters`` has one row per start.
    """
    def loss_one(parameters, x, labels, f):
        return classification_loss(parameters, x, labels, f, cfg, beta)

    value_and_gradient = jax.vmap(
        jax.value_and_grad(loss_one),
        in_axes=(0, None, None, None),
    )

    @jax.jit
    def run(initial_parameters, x, labels, f):
        parameters = jnp.asarray(initial_parameters)
        first_moment = jnp.zeros_like(parameters)
        second_moment = jnp.zeros_like(parameters)

        def body(step, state):
            parameters, first_moment, second_moment = state
            _, gradient = value_and_gradient(parameters, x, labels, f)

            norms = jnp.linalg.norm(gradient, axis=1, keepdims=True)
            gradient = gradient * jnp.minimum(
                1.0, gradient_clip / (norms + 1e-12)
            )

            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            t = step + 1
            first_corrected = first_moment / (1.0 - 0.9**t)
            second_corrected = second_moment / (1.0 - 0.999**t)

            fraction = step / jnp.maximum(1, n_steps - 1)
            rate = learning_rate * (
                0.1 + 0.9 * 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
            )
            parameters = parameters - rate * first_corrected / (
                jnp.sqrt(second_corrected) + 1e-8
            )
            return parameters, first_moment, second_moment

        parameters, _, _ = jax.lax.fori_loop(
            0,
            n_steps,
            body,
            (parameters, first_moment, second_moment),
        )
        losses = jax.vmap(loss_one, in_axes=(0, None, None, None))(
            parameters, x, labels, f
        )
        return parameters, losses

    return run


def make_value_and_grad(cfg, *, beta):
    """Jitted ``(w, x, labels, f) -> (loss, gradient)`` of the classification loss."""
    return jax.jit(
        jax.value_and_grad(
            lambda w, x, labels, f: classification_loss(
                w, x, labels, f, cfg, beta
            )
        )
    )


def balanced_accuracy(labels, predictions):
    """Mean of sensitivity and specificity (NaN if one class is absent)."""
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    positive = labels
    negative = ~labels
    if not np.any(positive) or not np.any(negative):
        return np.nan
    sensitivity = np.mean(predictions[positive])
    specificity = np.mean(~predictions[negative])
    return float(0.5 * (sensitivity + specificity))


def evaluate_classifier(free_parameters, x, labels, f, cfg, beta):
    """BCE, accuracy and balanced accuracy of a student on (x, labels)."""
    score = np.asarray(forward(cfg.expand(free_parameters), x, f, cfg))
    logits = beta * score
    labels = np.asarray(labels, dtype=float)
    predictions = logits > 0
    bce = float(
        np.mean(np.logaddexp(0.0, logits) - labels * logits)
    )
    return {
        "bce": bce,
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
    }


def _adam_candidates(
    cfg, f, x_train, y_train, *, adam_runner, seed, base_seed, n_restarts,
    extra_start,
):
    """Run Adam from ``n_restarts`` prior samples; return parameters and losses."""
    rng = rng_from(base_seed, 300, seed)
    initial = np.stack(
        [sample_free_prior(rng, cfg) for _ in range(n_restarts)]
    )
    if extra_start is not None:
        initial[0] = np.asarray(extra_start)

    adam_parameters, adam_losses = adam_runner(
        jnp.asarray(initial),
        jnp.asarray(x_train),
        jnp.asarray(y_train),
        jnp.asarray(f),
    )
    return np.asarray(adam_parameters), np.asarray(adam_losses)


def _polish_and_evaluate(
    value_and_jacobian, candidates, losses, cfg, f, x_train, y_train, x_test,
    y_test, *, beta, polish_top, polish_maxiter,
):
    """L-BFGS polish of the best Adam results, then train and test metrics."""
    best_index = int(np.argmin(losses))
    best_parameters = candidates[best_index].copy()
    best_loss = float(losses[best_index])

    polish_indices = np.argsort(losses)[:polish_top]
    polish_successes = 0
    for index in polish_indices:
        result = minimize(
            value_and_jacobian,
            candidates[index],
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": polish_maxiter,
                "ftol": 1e-12,
                "gtol": 1e-9,
            },
        )
        if result.success:
            polish_successes += 1
        if float(result.fun) < best_loss:
            best_loss = float(result.fun)
            best_parameters = np.asarray(result.x)

    train_metrics = evaluate_classifier(
        best_parameters, x_train, y_train, f, cfg, beta
    )
    test_metrics = evaluate_classifier(
        best_parameters, x_test, y_test, f, cfg, beta
    )
    return {
        "free_parameters": best_parameters,
        "train_bce": train_metrics["bce"],
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "test_bce": test_metrics["bce"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "polish_successes": int(polish_successes),
    }


def fit_classifier_local_jit(
    cfg,
    f,
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    seed,
    base_seed,
    beta,
    adam_runner,
    n_restarts,
    polish_top,
    polish_maxiter,
    extra_start=None,
):
    """Train one student; the L-BFGS objective is compiled per fit.

    Adam runs from ``n_restarts`` prior samples drawn from the stream
    ``(base_seed, 300, seed)`` (the first replaced by ``extra_start`` if
    given), then the ``polish_top`` best results are polished with L-BFGS.
    Returns the best free parameters, train and test metrics and the number
    of successful polishing runs.
    """
    candidates, losses = _adam_candidates(
        cfg, f, x_train, y_train, adam_runner=adam_runner, seed=seed,
        base_seed=base_seed, n_restarts=n_restarts, extra_start=extra_start,
    )

    x_jax = jnp.asarray(x_train)
    y_jax = jnp.asarray(y_train)
    f_jax = jnp.asarray(f)
    objective = jax.jit(
        jax.value_and_grad(
            lambda w: classification_loss(w, x_jax, y_jax, f_jax, cfg, beta)
        )
    )

    def value_and_jacobian(parameters):
        value, gradient = objective(jnp.asarray(parameters))
        return float(value), np.asarray(gradient, dtype=float)

    return _polish_and_evaluate(
        value_and_jacobian, candidates, losses, cfg, f, x_train, y_train,
        x_test, y_test, beta=beta, polish_top=polish_top,
        polish_maxiter=polish_maxiter,
    )


def fit_classifier_shared_jit(
    cfg,
    f,
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    seed,
    base_seed,
    beta,
    adam_runner,
    value_and_grad,
    n_restarts,
    polish_top,
    polish_maxiter,
    extra_start=None,
):
    """Train one student; the L-BFGS objective is the shared ``value_and_grad``.

    Identical to :func:`fit_classifier_local_jit` except that the polishing
    objective is the compiled function from :func:`make_value_and_grad`,
    which receives the training data as arguments.
    """
    candidates, losses = _adam_candidates(
        cfg, f, x_train, y_train, adam_runner=adam_runner, seed=seed,
        base_seed=base_seed, n_restarts=n_restarts, extra_start=extra_start,
    )

    x_jax = jnp.asarray(x_train)
    y_jax = jnp.asarray(y_train)
    f_jax = jnp.asarray(f)

    def value_and_jacobian(parameters):
        value, gradient = value_and_grad(
            jnp.asarray(parameters), x_jax, y_jax, f_jax
        )
        return float(value), np.asarray(gradient, dtype=float)

    return _polish_and_evaluate(
        value_and_jacobian, candidates, losses, cfg, f, x_train, y_train,
        x_test, y_test, beta=beta, polish_top=polish_top,
        polish_maxiter=polish_maxiter,
    )
