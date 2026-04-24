#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
"""
Speedcheck: scalar Python loop vs. prange-batched GERG-2008 properties.
"""
import os
import time
import numpy as np

from GasNetSim.components.gas_mixture.eos.gerg2008_numba import (
    PropertiesGERG_numba,
)
from GasNetSim.components.gas_mixture.eos.gerg2008_batch import (
    PropertiesGERG_batch_numba,
)


# Representative natural-gas composition (21 components, GERG order)
BASE_X = np.array([
    0.77824,  # methane
    0.02000,  # nitrogen
    0.06000,  # carbon dioxide
    0.08000,  # ethane
    0.03000,  # propane
    0.00150,  # isobutane
    0.00300,  # n-butane
    0.00050,  # isopentane
    0.00165,  # n-pentane
    0.00215,  # n-hexane
    0.00088,  # n-heptane
    0.00024,  # n-octane
    0.00015,  # n-nonane
    0.00009,  # n-decane
    0.00400,  # hydrogen
    0.00500,  # oxygen
    0.00200,  # carbon monoxide
    0.00010,  # water
    0.00250,  # hydrogen sulfide
    0.00700,  # helium
    0.00100,  # argon
], dtype=np.float64)
assert abs(BASE_X.sum() - 1.0) < 1e-6


def make_batch(N, seed=0):
    rng = np.random.default_rng(seed)
    # Small random perturbations around a reasonable gas-phase operating point
    T = rng.uniform(280.0, 320.0, size=N).astype(np.float64)          # K
    P = rng.uniform(3_000.0, 8_000.0, size=N).astype(np.float64)      # kPa (30-80 bar)
    x = np.tile(BASE_X, (N, 1))
    # Tiny composition jitter, renormalised
    x *= rng.uniform(0.98, 1.02, size=(N, 21))
    x /= x.sum(axis=1, keepdims=True)
    return T, P, x


def run_scalar_loop(T, P, x):
    N = T.shape[0]
    out = np.empty((N, 20), dtype=np.float64)
    for i in range(N):
        p = PropertiesGERG_numba(T[i], P[i], x[i])
        for j in range(20):
            out[i, j] = p[j]
    return out


def time_call(fn, *args, repeats=3):
    # Return best-of-N wall time in seconds
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    print(f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS', '(unset)')}")
    from numba import get_num_threads
    print(f"numba active threads: {get_num_threads()}")

    # Warmup: compile both paths on a tiny batch
    print("warming up (JIT compile)...")
    Tw, Pw, xw = make_batch(4)
    _ = run_scalar_loop(Tw, Pw, xw)
    _ = PropertiesGERG_batch_numba(Tw, Pw, xw)
    print("ready.\n")

    print(f"{'N':>8} | {'scalar (s)':>12} | {'batch (s)':>12} | {'speedup':>8} | "
          f"{'scalar µs/call':>16} | {'batch µs/call':>15}")
    print("-" * 90)

    for N in [100, 500, 1_000, 5_000, 10_000, 50_000]:
        T, P, x = make_batch(N)
        # Verify both paths agree before timing
        ref = run_scalar_loop(T[:8], P[:8], x[:8])
        got = PropertiesGERG_batch_numba(T[:8], P[:8], x[:8])
        assert np.allclose(ref, got, rtol=1e-10, atol=1e-10), \
            "scalar and batch results differ"

        # Fewer repeats for larger N to keep runtime reasonable
        repeats = 5 if N <= 1_000 else 3 if N <= 10_000 else 2
        t_scalar = time_call(run_scalar_loop, T, P, x, repeats=repeats)
        t_batch = time_call(PropertiesGERG_batch_numba, T, P, x, repeats=repeats)

        print(f"{N:>8d} | {t_scalar:>12.4f} | {t_batch:>12.4f} | "
              f"{t_scalar/t_batch:>7.2f}x | "
              f"{t_scalar/N*1e6:>16.2f} | {t_batch/N*1e6:>15.2f}")


if __name__ == "__main__":
    main()
