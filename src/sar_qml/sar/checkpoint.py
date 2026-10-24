"""Configuration hash and resumable per-task result CSV.

Every completed fit is appended to a CSV as one row keyed by ``task_id``, so
an interrupted run resumes where it stopped and a finished run only reloads
its results. The file name contains ``RUN_ID``, a hash of the full run
configuration, so that incompatible runs are never combined.
"""

import hashlib
import json

import numpy as np
import pandas as pd


def config_hash(spec):
    """First 12 hex digits of the SHA-256 of the sorted JSON of ``spec``."""
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def completed_tasks(path):
    """Set of task IDs already stored in the CSV at ``path``."""
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["task_id"].astype(str))


def append_result(path, row, columns):
    """Append one row to the CSV, filling missing ``columns`` with NaN."""
    complete_row = {column: row.get(column, np.nan) for column in columns}
    pd.DataFrame([complete_row], columns=columns).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


class TaskLog:
    """The result CSV of one run together with its set of completed tasks.

    ``task_id in log`` tells whether a task is done; ``log.record(task_id,
    row)`` appends the row (with ``run_id`` and ``task_id`` added) and marks
    the task as done.
    """

    def __init__(self, path, columns, run_id):
        self.path = path
        self.columns = list(columns)
        self.run_id = run_id
        self.completed = completed_tasks(path)

    def __contains__(self, task_id):
        return task_id in self.completed

    def __len__(self):
        return len(self.completed)

    def record(self, task_id, row):
        append_result(
            self.path, {"run_id": self.run_id, "task_id": task_id, **row}, self.columns
        )
        self.completed.add(task_id)
