"""Seeds, balanced labels, balanced accuracy and check printing."""

import jax
import jax.numpy as jnp
import numpy as np


def seed_from(base_seed, *parts):
    """32-bit seed derived from ``base_seed`` and a tuple of integers."""
    sequence = np.random.SeedSequence([base_seed, *[int(p) for p in parts]])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def balanced_labels(key, batch_size, n_classes):
    """Labels in which every consecutive block of ``n_classes`` is a permutation.

    Every prefix whose length is a multiple of ``n_classes`` is therefore
    exactly balanced, not only the full batch.
    """
    if batch_size % n_classes:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by Q={n_classes}"
        )
    scores = jax.random.uniform(key, (batch_size // n_classes, n_classes))
    return jnp.argsort(scores, axis=1).reshape(-1)


def balanced_accuracy(labels, predictions, n_classes):
    """Mean over classes of the per-class recall."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    return float(
        np.mean(
            [np.mean(predictions[labels == cls] == cls) for cls in range(n_classes)]
        )
    )


def print_checks(validity, width):
    """Print a dict of validity checks as an aligned two-column table."""
    for key, value in validity.items():
        if isinstance(value, (bool, np.bool_)):
            print(f"{key:{width}s} {value}")
        else:
            print(f"{key:{width}s} {value:.6g}")
    print("all validity gates passed")
