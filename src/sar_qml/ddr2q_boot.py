"""Device bootstrap -- MUST be imported before jax.

XLA reads its flags once, at first jax import, so the number of host devices has
to be fixed before anything else touches jax.  Splitting the 16 physical cores
into 8 logical devices (2 threads each) measured ~3x faster than 1 device x 16
threads for this workload; 16 devices deadlocks in the all-reduce rendezvous.

Usage (first two lines of an experiment script):

    from sar_qml.ddr2q_boot import init_devices
    init_devices()
"""

from __future__ import annotations

import os

N_DEVICES = int(os.environ.get("DDR2Q_DEVICES", "8"))


def init_devices(n: int | None = None) -> int:
    n = N_DEVICES if n is None else n
    if "jax" in __import__("sys").modules:
        raise RuntimeError("init_devices() must be called before importing jax")
    flags = os.environ.get("XLA_FLAGS", "")
    if "host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (
            flags + f" --xla_force_host_platform_device_count={n}").strip()
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, 16 // n)))
    return n
