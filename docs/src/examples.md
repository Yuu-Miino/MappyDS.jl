# Working examples

The `examples/` directory of the
[repository](https://github.com/Yuu-Miino/MappyDS.jl/tree/main/examples)
contains one [Literate.jl](https://fredrikekre.github.io/Literate.jl/) script
per model below; the pages of this section are generated from them and show
the output of every step. The scripts are not part of the installed package.
To run one yourself, install MappyDS together with a Makie backend and
StaticArrays in any environment and include the script:

```julia
using Pkg
Pkg.add(["MappyDS", "CairoMakie", "StaticArrays"])
include("examples/duffing.jl")
```

Each script builds the system, draws an orbit (with `CairoMakie`; the
interactive viewer with `GLMakie` is shown in a non-executed block) and runs
the numerical analysis: periodic points, their continuation, local
bifurcations and bifurcation curves. They mirror the scripts of the Python
package (see [Python package](@ref)); the mode names follow the Python
scripts.

| Example | What it shows |
| --- | --- |
| [Hénon map](@ref) | A discrete map as a `Diffeomorphism`; the strange attractor, a bifurcation diagram and the maximum Lyapunov exponent from the Jacobian |
| [Three coupled logistic maps](@ref) | Fixed-point continuation in ``\alpha``, multipliers along the branch, period-doubling curve in ``(\alpha, \beta)`` |
| [Izhikevich neuron](@ref) | A 1-D Poincaré map with a firing reset; period doubling in ``(a, I)`` |
| [Forced Duffing oscillator](@ref) | Stroboscopic map of a forced oscillator; the period doublings at ``B \approx 0.171687`` and ``0.368655`` |
| [Forced Alpazur oscillator](@ref) | A piecewise-smooth system with hysteresis and two end modes; period doubling in ``(V, B_1)`` |

The hybrid-system examples reproduce figures of [miino2024](@citet).
