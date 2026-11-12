"""Per-operation timings of mappy-ds (Python) on the benchmark systems.

Mirrors bench.jl: the same systems, tolerances and operation counts, so that
the per-call times are comparable. Writes a JSON file (argv[1]) with
{task: {"per_call_ms": float, "n": int}}.

Run: python bench.py out.json [scale]   (scale < 1 shortens the batches)
"""

import json
import statistics
import sys
import time

import numpy as np
import sympy

from mappy_ds import (
    ContinuousMode,
    Diffeomorphism,
    DiscreteMode,
    PoincareMap,
    border,
    mapping,
    vector_field,
)
from mappy_ds.root import find_cycle, find_local_bf, trace_cycle, trace_local_bf

SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
REPEATS = 3


def n_of(n: int) -> int:
    return max(1, int(round(n * SCALE)))


# -- systems (as in the test suites of both packages) --------------------------


@vector_field(dim=2)
def izhikevich(y, p):
    v, u = y
    return np.array([0.04 * v**2 + 5 * v + 140 - u + p["I"], p["a"] * (p["b"] * v - u)])


@border(dim=2)
def fire(y, _):
    return y[0] - 30


@mapping(2)
def firing(y, p):
    return y + np.array([p["c"] - 30, p["d"]])


@mapping(2, 1)
def reduce_u(y, _):
    return np.array([0, 1]) @ y


@mapping(1, 2)
def revert(u, _):
    return u * np.array([0, 1]) + np.array([30, 0])


def make_izhikevich_map() -> PoincareMap:
    return PoincareMap(
        [
            ContinuousMode("A", izhikevich, borders=[fire], rtol=1e-6, atol=1e-9),
            DiscreteMode("B", firing),
            DiscreteMode("P", reduce_u),
            DiscreteMode("Pinv", revert),
        ],
        {"A": ["P"], "B": "A", "P": "Pinv", "Pinv": "B"},
        initial_mode="Pinv",
    )


@mapping(2)
def henon(y, p):
    x, u = y
    return np.array([1 - p["a"] * x**2 + u, p["b"] * x])


@vector_field(3)
def duffing(y, p):
    x, v, t = y
    return np.array([v, -p["k"] * v - x**3 + p["B0"] + p["B"] * sympy.cos(t), 1])


@border(3)
def duffing_section(y, _):
    return y[2] - 2 * sympy.pi


@mapping(3)
def duffing_t_reset(y, _):
    return y - np.array([0, 0, 2 * sympy.pi])


@mapping(2, 3)
def append_time(y, _):
    return np.append(y, 0)


@mapping(3, 2)
def drop_time(y, _):
    return y[0:2]


def make_duffing_map() -> PoincareMap:
    return PoincareMap(
        [
            ContinuousMode(
                "duffing", duffing, borders=[duffing_section], rtol=1e-6, atol=1e-9
            ),
            DiscreteMode("t_reset", duffing_t_reset),
            DiscreteMode("p", append_time),
            DiscreteMode("pinv", drop_time),
        ],
        {"p": "duffing", "duffing": ["t_reset"], "t_reset": "pinv", "pinv": "p"},
        "p",
    )


IZH = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}
IZH8 = {**IZH, "I": 8.0}
HEN = {"a": 1.4, "b": 0.3}
DUF = {"k": 0.2, "B0": 0.1, "B": 0.1}
DUF_Y0 = np.array([0.232147751858, 0.075591687407])

results: dict[str, dict] = {}


def bench(name: str, n: int, fn) -> None:
    """Time `fn()` called n times; report the median per-call time of REPEATS batches."""
    fn()  # warm-up (caches, first-call effects)
    per_call = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        per_call.append((time.perf_counter() - t0) / n * 1e3)
    results[name] = {"per_call_ms": statistics.median(per_call), "n": n}
    print(f"{name:28s} {results[name]['per_call_ms']:12.4f} ms  (n={n})", flush=True)


hmap = Diffeomorphism(henon)
izh = make_izhikevich_map()
duf = make_duffing_map()
y_h = np.array([0.1, 0.1])

# Discrete map: one evaluation
bench("henon_image", n_of(20000), lambda: hmap.image_detail(y_h, HEN, jacobian=False))
bench("henon_jac_hes", n_of(5000), lambda: hmap.image_detail(y_h, HEN, hessian=True))
bench(
    "henon_full",
    n_of(2000),
    lambda: hmap.image_detail(y_h, HEN, hessian=True, parameter_jacobian=True, parameter_hessian=True),
)

# Poincaré map of the Izhikevich neuron (1-D section, one firing per iteration)
bench("izh_image", n_of(300), lambda: izh.image_detail(-3.0, IZH, jacobian=False))
bench("izh_jac", n_of(300), lambda: izh.image_detail(-3.0, IZH))
bench("izh_jac_hes", n_of(200), lambda: izh.image_detail(-3.0, IZH, hessian=True))
bench(
    "izh_full",
    n_of(100),
    lambda: izh.image_detail(-3.0, IZH, hessian=True, parameter_jacobian=True, parameter_hessian=True),
)

# Stroboscopic map of the forced Duffing oscillator (one period 2π)
bench("duffing_image", n_of(100), lambda: duf.image_detail(DUF_Y0, DUF, jacobian=False))
bench("duffing_jac_hes", n_of(60), lambda: duf.image_detail(DUF_Y0, DUF, hessian=True))
bench(
    "duffing_full",
    n_of(30),
    lambda: duf.image_detail(DUF_Y0, DUF, hessian=True, parameter_jacobian=True, parameter_hessian=True),
)

# Periodic points, continuation, local bifurcations
bench("izh_find_cycle", n_of(20), lambda: find_cycle(izh, -4.6, IZH8))
fp = find_cycle(izh, -4.6, IZH8).y
n_trace = 20  # fixed: fewer points would make the continuation steps too large
bench(
    "izh_trace_cycle_per_point",
    1,
    lambda: trace_cycle(izh, fp, IZH8, "I", 4.5, resolution=n_trace),
)
results["izh_trace_cycle_per_point"]["per_call_ms"] /= n_trace
results["izh_trace_cycle_per_point"]["n"] = n_trace

duf_steps = trace_cycle(duf, find_cycle(duf, DUF_Y0, DUF).y, DUF, "B", 0.17, resolution=10)
last = duf_steps[-1]
bench("duffing_find_local_bf", n_of(5), lambda: find_local_bf(duf, last.y, last.params, "B", np.pi))

hp = {"a": 0.35, "b": 0.3}
hfp = find_cycle(hmap, np.array([0.9, 0.3]), hp).y
hbf = find_local_bf(hmap, hfp, hp, "a", np.pi)
n_bf = 100
bench(
    "henon_trace_local_bf_per_point",
    1,
    lambda: trace_local_bf(hmap, hbf.y, hbf.params, "a", hbf.theta, "b", 0.2, resolution=n_bf),
)
results["henon_trace_local_bf_per_point"]["per_call_ms"] /= n_bf
results["henon_trace_local_bf_per_point"]["n"] = n_bf

with open(sys.argv[1], "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {sys.argv[1]}")
