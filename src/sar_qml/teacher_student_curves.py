"""Learning-curve experiment for the teacher-student classification task.

Stage A (score realizability) fits the continuous teacher score on a dense
grid with every student arm by trust-region least squares. Stage B
(classification) trains every arm on hard teacher labels for several
training-set sizes and folds. The module also holds the teacher-level
bootstrap used for the learning curves and the JSON results cache.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .teacher_student import (
    forward,
    paired_seed,
    rng_from,
    sample_free_prior,
    teacher_test_set,
    teacher_training_pool,
)

# ---------------------------------------------------------------------------
# Stage A: score-realizability diagnostic
# ---------------------------------------------------------------------------


def score_fit(cfg, f, x, target, starts, *, max_nfev):
    """Best least-squares fit of a target score over several starts.

    Returns the best mean squared error ``"mse"`` and the corresponding error
    relative to the mean squared target ``"relative_mse"``.
    """
    x_jax = jnp.asarray(x)
    f_jax = jnp.asarray(f)
    target_jax = jnp.asarray(target)

    prediction = lambda w: forward(cfg.expand(w), x_jax, f_jax, cfg)
    residual = jax.jit(lambda w: prediction(w) - target_jax)
    jacobian = jax.jit(jax.jacfwd(prediction))

    best_loss = np.inf
    best_parameters = None
    for start in starts:
        result = least_squares(
            lambda w: np.asarray(residual(jnp.asarray(w))),
            np.asarray(start),
            jac=lambda w: np.asarray(jacobian(jnp.asarray(w))),
            method="trf",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=max_nfev,
        )
        loss = float(np.mean(result.fun**2))
        if loss < best_loss:
            best_loss = loss
            best_parameters = np.asarray(result.x)

    prediction_best = np.asarray(
        forward(cfg.expand(best_parameters), x, f, cfg)
    )
    relative_mse = float(
        np.mean((prediction_best - target) ** 2)
        / np.mean(np.asarray(target) ** 2)
    )
    return {
        "mse": best_loss,
        "relative_mse": relative_mse,
    }


def diagnostic_grid(n_qubits, points_per_axis=15):
    """Regular points_per_axis x points_per_axis grid on [0, 2 pi)^2."""
    if n_qubits != 2:
        raise ValueError(
            "the dense Cartesian diagnostic grid is intentionally limited "
            "to the reported two-qubit experiment"
        )
    grid = 2 * np.pi * np.arange(points_per_axis) / points_per_axis
    x0, x1 = np.meshgrid(grid, grid, indexing="ij")
    return np.stack([x0.ravel(), x1.ravel()], axis=1)


def run_score_diagnostic(
    cfg, teachers, arms, f_star, *, n_restarts, max_nfev, base_seed
):
    """Fit every teacher score with every arm on the diagnostic grid.

    The starts are the teacher's own parameters plus ``n_restarts - 1``
    prior samples, shared by all arms. Returns one row per teacher and arm.
    """
    x = diagnostic_grid(cfg.n_qubits)
    rows = []
    for teacher in teachers:
        target = np.asarray(
            forward(teacher["full_parameters"], x, f_star, cfg)
        )
        start_rng = rng_from(base_seed, 200, teacher["teacher"])
        starts = [teacher["free_parameters"].copy()]
        starts.extend(
            sample_free_prior(start_rng, cfg)
            for _ in range(n_restarts - 1)
        )
        for arm, f in arms.items():
            fit = score_fit(cfg, f, x, target, starts, max_nfev=max_nfev)
            rows.append({"teacher": teacher["teacher"], "arm": arm, **fit})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage B: classification learning curves
# ---------------------------------------------------------------------------


def run_classification_experiment(
    cfg,
    teachers,
    arms,
    f_star,
    fit,
    *,
    n_grid,
    n_folds,
    n_test,
    base_seed,
    warm_start_across_n,
):
    """Train all arms for every teacher, fold and training-set size.

    ``fit(f, x_train, y_train, x_test, y_test, seed=..., extra_start=...)``
    trains one student. Each fold draws a training pool of ``max(n_grid)``
    points whose prefixes are the training sets; all arms share the data and
    the initial-parameter seed. Prints one line per teacher, fold and size and
    returns one row per fit (without the fitted parameters).
    """
    rows = []
    n_max = max(n_grid)

    for teacher in teachers:
        teacher_id = teacher["teacher"]
        x_test, y_test = teacher_test_set(
            teacher, f_star, cfg, n_test=n_test, base_seed=base_seed
        )

        for fold in range(n_folds):
            # Warm starts are reset for every independent fold.
            warm = {}
            x_pool, y_pool = teacher_training_pool(
                teacher, fold, f_star, cfg, n_points=n_max, base_seed=base_seed
            )

            for n in sorted(n_grid):
                x_train = x_pool[:n]
                y_train = y_pool[:n]

                message = []
                for arm, f in arms.items():
                    result = fit(
                        f,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        seed=paired_seed(teacher_id, fold, n),
                        extra_start=(
                            warm.get(arm) if warm_start_across_n else None
                        ),
                    )
                    if warm_start_across_n:
                        warm[arm] = result["free_parameters"]

                    rows.append(
                        {
                            "teacher": int(teacher_id),
                            "fold": int(fold),
                            "n": int(n),
                            "arm": arm,
                            **{
                                key: value
                                for key, value in result.items()
                                if key != "free_parameters"
                            },
                        }
                    )
                    message.append(
                        f"{arm}:{result['test_balanced_accuracy']:.3f}"
                    )

                print(
                    f"teacher={teacher_id:02d} fold={fold:02d} n={n:03d} | "
                    + " ".join(message),
                    flush=True,
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Teacher-level bootstrap
# ---------------------------------------------------------------------------


def teacher_level_table(results, metric):
    """Average the metric over folds within each teacher, arm and N."""
    return (
        results.groupby(["teacher", "arm", "n"], as_index=False)[metric]
        .mean()
        .sort_values(["teacher", "arm", "n"])
    )


def paired_teacher_bootstrap(
    results, metric, arms, *, n_bootstrap, base_seed, seed=0
):
    """Teacher-level mean and 95% interval from resampling teachers.

    All arms and sample sizes of a teacher are resampled together. Returns
    the teachers, arms and sample sizes together with ``estimate``,
    ``lower``, ``upper`` and the bootstrap draws, each of shape
    (n_arms, n_sizes) (draws with a leading bootstrap axis).
    """
    table = teacher_level_table(results, metric)
    teachers = np.array(sorted(table["teacher"].unique()))
    arms = list(arms)
    sample_sizes = sorted(table["n"].unique())

    values = np.empty((len(teachers), len(arms), len(sample_sizes)))
    for i, teacher in enumerate(teachers):
        for j, arm in enumerate(arms):
            part = table[
                (table["teacher"] == teacher) & (table["arm"] == arm)
            ].set_index("n")
            values[i, j] = part.loc[sample_sizes, metric]

    estimate = values.mean(axis=0)
    rng = rng_from(base_seed, 500, seed)
    bootstrap = np.empty(
        (n_bootstrap, len(arms), len(sample_sizes))
    )
    for draw in range(n_bootstrap):
        indices = rng.integers(0, len(teachers), len(teachers))
        bootstrap[draw] = values[indices].mean(axis=0)

    lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {
        "teachers": teachers,
        "arms": arms,
        "n": sample_sizes,
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "bootstrap": bootstrap,
    }


def summary_table(summary):
    """Long table (arm, n, mean, lower95, upper95) of a bootstrap summary."""
    rows = []
    for arm_index, arm in enumerate(summary["arms"]):
        for n_index, n in enumerate(summary["n"]):
            rows.append(
                {
                    "arm": arm,
                    "n": n,
                    "mean": summary["estimate"][arm_index, n_index],
                    "lower95": summary["lower"][arm_index, n_index],
                    "upper95": summary["upper"][arm_index, n_index],
                }
            )
    return pd.DataFrame(rows)


def paired_difference_at_largest_n(
    results, left, right, metric, *, n_bootstrap, base_seed
):
    """Paired teacher-bootstrap difference between two arms at the largest N."""
    n = max(results["n"])
    table = teacher_level_table(results, metric)
    pivot = (
        table[table["n"] == n]
        .pivot(index="teacher", columns="arm", values=metric)
    )
    difference = np.asarray(pivot[left] - pivot[right])
    rng = rng_from(base_seed, 501, len(right))
    bootstrap = np.array(
        [
            np.mean(
                difference[
                    rng.integers(0, len(difference), len(difference))
                ]
            )
            for _ in range(n_bootstrap)
        ]
    )
    return {
        "n": int(n),
        "mean": float(np.mean(difference)),
        "lower95": float(np.quantile(bootstrap, 0.025)),
        "upper95": float(np.quantile(bootstrap, 0.975)),
    }


# ---------------------------------------------------------------------------
# Results cache
# ---------------------------------------------------------------------------


def json_safe(value):
    """Convert NumPy containers and scalars to plain JSON types."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def teacher_records(teachers):
    """JSON-ready teacher records (without the full parameter tensor)."""
    return json_safe(
        [
            {
                "teacher": teacher["teacher"],
                "draws": teacher["draws"],
                "free_parameters": teacher["free_parameters"],
                "balance": teacher["balance"],
                "nonseparability": teacher["nonseparability"],
                "boundary_mass_005": teacher["boundary_mass_005"],
            }
            for teacher in teachers
        ]
    )


def load_cache(path, config, records):
    """Load cached results if ``path`` exists, else return None.

    Raises ValueError if the cache was produced with a different
    configuration and AssertionError if its teachers differ from ``records``.
    """
    if not path.exists():
        print(f"no cached results at {path}; Stages A and B will run")
        return None
    cached = json.loads(path.read_text())
    if cached["config"] != config:
        raise ValueError(
            f"{path} was produced with a different configuration; "
            "delete it to recompute."
        )
    for new, old in zip(records, cached["teachers"], strict=True):
        assert new["draws"] == old["draws"]
        assert np.allclose(new["free_parameters"], old["free_parameters"])
    print(f"loaded cached results from {path}")
    return cached


def save_results(path, config, records, score_results, classification_results,
                 classification_summary):
    """Write configuration, teachers and the results of both stages to JSON."""
    payload = {
        "config": config,
        "teachers": records,
        "score_results": score_results.to_dict(orient="records"),
        "classification_results": classification_results.to_dict(
            orient="records"
        ),
        "classification_summary": classification_summary.to_dict(
            orient="records"
        ),
    }
    path.write_text(json.dumps(json_safe(payload), indent=2))
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def angle_axis(axis, show_y=True):
    """Format an input-plane panel: [0, 2 pi]^2, equal aspect, ticks at 0, pi, 2 pi."""
    ticks = [0.0, np.pi, 2 * np.pi]
    labels = [r"$0$", r"$\pi$", r"$2\pi$"]
    axis.set_xlim(0.0, 2 * np.pi)
    axis.set_ylim(0.0, 2 * np.pi)
    axis.set_aspect("equal")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_yticks(ticks)
    axis.set_yticklabels(labels if show_y else [])
    axis.set_xlabel(r"$x_0$")
    if show_y:
        axis.set_ylabel(r"$x_1$")
    axis.tick_params(which="both", top=False, right=False)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
