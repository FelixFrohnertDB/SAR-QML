"""Query-value sensitivity scan for the teacher-student classification task.

Students are retrained from scratch for many fixed query values: a global
scan along ``f(lambda) = lambda f*`` and a local scan of small mismatches
around ``f*``. Raw fits are checkpointed row by row in a CSV file tagged with
a hash of the full numerical specification (the run identifier), so an
interrupted run resumes where it stopped. The module also holds the
teacher-bootstrap aggregation and the tolerance estimate.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from .teacher_student import (
    paired_seed,
    rng_from,
    teacher_test_set,
    teacher_training_pool,
)

# ---------------------------------------------------------------------------
# Candidate queries
# ---------------------------------------------------------------------------


def wrap_angles(f):
    """Wrap angles to [-pi, pi)."""
    f = np.asarray(f, dtype=float)
    return (f + np.pi) % (2 * np.pi) - np.pi


def make_query_table(
    f_star,
    *,
    global_points,
    local_delta_min,
    local_delta_max,
    local_deltas,
    local_directions,
):
    """Table of all global and local candidate queries.

    The global scan has ``global_points`` values of lambda in [0, 2] plus
    exactly 0, 1 and 2. The local scan has ``local_deltas`` log-spaced radii
    times ``local_directions`` uniformly spaced directions around ``f_star``.
    """
    rows = []

    # linspace contains lambda = 1 for odd grid sizes; adding it explicitly keeps
    # the matched baseline in any grid.
    lambdas = np.unique(
        np.concatenate(
            [np.linspace(0.0, 2.0, global_points), [0.0, 1.0, 2.0]]
        )
    )
    for index, lam in enumerate(lambdas):
        f = lam * f_star
        rows.append(
            dict(
                scan="global",
                query_id=f"g{index:03d}",
                candidate=index,
                query_scale=float(lam),
                delta=np.nan,
                direction=np.nan,
                phi=np.nan,
                f0=float(f[0]),
                f1=float(f[1]),
            )
        )

    deltas = np.geomspace(local_delta_min, local_delta_max, local_deltas)
    phis = 2 * np.pi * np.arange(local_directions) / local_directions
    candidate = 0
    for delta_index, delta in enumerate(deltas):
        for direction, phi in enumerate(phis):
            displacement = delta * np.array([np.cos(phi), np.sin(phi)])
            f = wrap_angles(f_star + displacement)
            rows.append(
                dict(
                    scan="local",
                    query_id=f"l{delta_index:03d}d{direction:02d}",
                    candidate=candidate,
                    query_scale=np.nan,
                    delta=float(delta),
                    direction=int(direction),
                    phi=float(phi),
                    f0=float(f[0]),
                    f1=float(f[1]),
                )
            )
            candidate += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run identifier and checkpointing
# ---------------------------------------------------------------------------


def compute_run_id(run_spec):
    """First 12 hex digits of the SHA-256 of the sorted-key JSON specification."""
    return hashlib.sha256(
        json.dumps(run_spec, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def completed_keys(path):
    """(teacher, fold, query_id) keys already present in the checkpoint file."""
    if not path.exists():
        return set()
    previous = pd.read_csv(path)
    return set(
        zip(
            previous["teacher"].astype(int),
            previous["fold"].astype(int),
            previous["query_id"].astype(str),
        )
    )


def append_checkpoint(row, path):
    """Append one result row to the checkpoint file."""
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def run_query_sensitivity(
    cfg,
    teachers,
    queries,
    f_star,
    fit,
    *,
    result_path,
    run_id,
    n_folds,
    n_train,
    n_test,
    base_seed,
):
    """Train one student per teacher, fold and query, skipping completed fits.

    ``fit(f, x_train, y_train, x_test, y_test, seed=..., extra_start=...)``
    trains one student. For a teacher the test set is shared by all folds
    and queries; for a teacher and fold the training set and the initial
    parameters are shared by all queries. Every fit is appended to
    ``result_path`` as soon as it finishes. Returns the full checkpoint file.
    """
    completed = completed_keys(result_path)
    total = len(teachers) * n_folds * len(queries)
    done = len(completed)

    for teacher in teachers:
        teacher_id = int(teacher["teacher"])
        x_test, y_test = teacher_test_set(
            teacher, f_star, cfg, n_test=n_test, base_seed=base_seed
        )

        for fold in range(n_folds):
            x_train, y_train = teacher_training_pool(
                teacher, fold, f_star, cfg, n_points=n_train,
                base_seed=base_seed,
            )

            # The seed does not depend on the query, so every query starts
            # from the same initial parameter ensemble.
            seed = paired_seed(teacher_id, fold, n_train)

            for query in queries.itertuples(index=False):
                key = (teacher_id, fold, str(query.query_id))
                if key in completed:
                    continue

                f = np.array([query.f0, query.f1], dtype=float)
                result = fit(
                    f,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed=seed,
                    extra_start=None,
                )
                row = {
                    "run_id": run_id,
                    "teacher": teacher_id,
                    "fold": int(fold),
                    "n_train": int(n_train),
                    "scan": query.scan,
                    "query_id": query.query_id,
                    "candidate": int(query.candidate),
                    "query_scale": query.query_scale,
                    "delta": query.delta,
                    "direction": query.direction,
                    "phi": query.phi,
                    "f0": query.f0,
                    "f1": query.f1,
                    **{
                        name: value
                        for name, value in result.items()
                        if name != "free_parameters"
                    },
                }
                append_checkpoint(row, result_path)
                completed.add(key)
                done += 1
                print(
                    f"{done:05d}/{total:05d} | teacher={teacher_id:02d} "
                    f"fold={fold:02d} {query.query_id} | "
                    f"bal.acc={result['test_balanced_accuracy']:.4f}",
                    flush=True,
                )

    return pd.read_csv(result_path)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def fold_average(results, group_columns, metric):
    """Average the metric over folds within each teacher and group."""
    return (
        results.groupby(["teacher", *group_columns], as_index=False)[metric]
        .mean()
        .sort_values(["teacher", *group_columns])
    )


def bootstrap_mean_matrix(values, n_bootstrap, seed, *, base_seed):
    """Mean over teachers (rows) with a 95% teacher-bootstrap interval.

    Returns ``mean``, ``lower``, ``upper`` (one entry per column) and the
    bootstrap ``draws``.
    """
    values = np.asarray(values, dtype=float)
    rng = rng_from(base_seed, 500, seed)
    indices = rng.integers(
        0, values.shape[0], (n_bootstrap, values.shape[0])
    )
    draws = values[indices].mean(axis=1)
    return {
        "mean": values.mean(axis=0),
        "lower": np.quantile(draws, 0.025, axis=0),
        "upper": np.quantile(draws, 0.975, axis=0),
        "draws": draws,
    }


def first_crossing_tolerance(deltas, curve, baseline, drop):
    """Mismatch at which a nonincreasing envelope of the curve falls below baseline - drop.

    The crossing is interpolated linearly in log10(delta). Returns
    ``(delta, status)`` with status ``"resolved"``, ``"lower_bound"`` (no
    crossing within the sampled range) or ``"upper_bound"`` (already below at
    the smallest delta).
    """
    deltas = np.asarray(deltas, dtype=float)
    curve = np.asarray(curve, dtype=float)
    threshold = baseline - drop
    monotone = np.minimum.accumulate(np.concatenate([[baseline], curve]))[1:]
    below = np.flatnonzero(monotone < threshold)
    if len(below) == 0:
        return float(deltas[-1]), "lower_bound"
    index = int(below[0])
    if index == 0:
        return float(deltas[0]), "upper_bound"
    x0, x1 = np.log10(deltas[index - 1 : index + 1])
    y0, y1 = monotone[index - 1 : index + 1]
    if np.isclose(y0, y1):
        return float(deltas[index - 1]), "resolved"
    fraction = np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0)
    return float(10 ** (x0 + fraction * (x1 - x0))), "resolved"


def bootstrap_tolerance(deltas, worst_matrix, baseline_matrix, drop, *,
                        n_bootstrap, base_seed):
    """Paired teacher bootstrap of the tolerance.

    In every draw the same resampled teachers (rows of ``worst_matrix`` and
    entries of ``baseline_matrix``) determine the baseline and the
    worst-direction curve. Returns the tolerance values and their statuses.
    """
    rng = rng_from(base_seed, 504)
    n_teachers = len(baseline_matrix)
    values = []
    statuses = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_teachers, n_teachers)
        value, status = first_crossing_tolerance(
            deltas,
            worst_matrix[indices].mean(axis=0),
            float(baseline_matrix[indices].mean()),
            drop,
        )
        values.append(value)
        statuses.append(status)
    return values, statuses
