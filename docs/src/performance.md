# Performance

The Julia and the Python implementation compute the same quantities with the
same algorithms (variational equations with the analytic border correction,
Newton-type solvers with analytic Jacobians), so the difference between them
is the cost of the language runtime: Julia compiles the user's functions and
the variational systems into native code, while `mappy-ds` evaluates
SymPy-generated functions and small NumPy arrays from Python.

The numbers below are **indicative only**. They were measured on one
GitHub-hosted runner (`ubuntu-latest`, Intel Xeon Platinum 8573C, 4 cores) on
2026-11-12 with MappyDS.jl 0.3.0 on Julia 1.13 and `mappy-ds` 0.3.0 on
Python 3.12 (NumPy 2.5, SciPy 1.18, SymPy 1.14). Absolute times vary with the
machine; the ratios are more stable.

## Setup

Both benchmarks build the same systems with the same tolerances
(`reltol = 1e-6`, `abstol = 1e-9`; `RK45` in Python, `Tsit5` in Julia):

- the Hénon map as a [`Diffeomorphism`](@ref) (a discrete map, no ODE);
- the Izhikevich neuron as a one-dimensional [`PoincareMap`](@ref) with one
  firing per iteration;
- the forced Duffing oscillator as a stroboscopic map (one period ``2\pi``).

Each operation is called `n` times after one warm-up call and the per-call
time is the median of three such batches. For Julia the first call, which
includes compilation, is listed separately.

## Per-operation times

| Operation | n | Python (mappy-ds) ms | Julia (MappyDS.jl) ms | Speed-up | Julia first call ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hénon, image only | 20000 | 0.0214 | 0.0011 | ×20 | 610 |
| Hénon, Jacobian + Hessian | 5000 | 0.0288 | 0.0016 | ×18 | 29 |
| Hénon, + parameter and mixed derivatives | 2000 | 0.0979 | 0.0017 | ×58 | 29 |
| Izhikevich map, image only | 300 | 2.14 | 0.076 | ×28 | 4814 |
| Izhikevich map, Jacobian | 300 | 4.04 | 0.100 | ×40 | 73 |
| Izhikevich map, Jacobian + Hessian | 200 | 7.11 | 0.130 | ×55 | 353 |
| Izhikevich map, + `jac_p` + `hes_p` | 100 | 31.8 | 0.178 | ×178 | 762 |
| Duffing map, image only | 100 | 2.34 | 0.095 | ×25 | 1871 |
| Duffing map, Jacobian + Hessian | 60 | 9.77 | 0.212 | ×46 | 257 |
| Duffing map, + `jac_p` + `hes_p` | 30 | 37.8 | 0.319 | ×119 | 441 |
| `find_cycle` (Izhikevich, one solve) | 20 | 24.4 | 1.77 | ×14 | 2412 |
| `trace_cycle` (Izhikevich, per point) | 20 | 29.4 | 1.68 | ×18 | 258 |
| `find_local_bf` (Duffing, one solve) | 5 | 240 | 1.71 | ×141 | 1048 |
| `trace_local_bf` (Hénon, per point) | 100 | 1.05 | 0.042 | ×25 | 492 |

## Reading the table

- In steady state the Julia implementation is 14 to 180 times faster. A
  plain Poincaré-map evaluation is 25 to 30 times faster; the gap widens with
  the order of the derivatives, because the Python implementation assembles
  the right-hand side of the variational equations from several small array
  operations per solver step, whose interpreter overhead dominates.
- The largest ratio, for [`find_local_bf`](@ref), combines the cost of the
  mixed derivatives `hes_p` with the Newton iterations of the bifurcation
  system.
- Julia pays for this with compilation on the first call: about five
  seconds for the first Poincaré-map evaluation of a session and about two
  seconds for the first `find_cycle`. For one or two interactive evaluations
  the Python package finishes sooner; for parameter sweeps, continuation and
  bifurcation curves, where the same operation is repeated hundreds of
  times, the Julia package is the faster choice.

## Reproducing

The benchmark scripts are in the `bench/` directory of the repository:
`bench/bench.py` (Python), `bench/bench.jl` (Julia, environment
`bench/Project.toml`) and `bench/summarize.py` (comparison table). The
manual GitHub Actions workflow *Benchmark* runs both on one runner and writes
the table to the job summary; a scale factor shortens the batches.

```sh
# Python (from a checkout of mappy with its environment)
python bench/bench.py bench_python.json

# Julia (from a checkout of MappyDS.jl)
julia --project=bench -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=bench bench/bench.jl bench_julia.json

python bench/summarize.py bench_python.json bench_julia.json
```
