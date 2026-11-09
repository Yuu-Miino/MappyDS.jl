# Examples

Literate.jl scripts, one per model. They are rendered, with their output, in
the "Working examples" section of the
[documentation](https://yuu-miino.github.io/doc.mappy/julia/stable/examples/),
and mirror the scripts of the Python package `mappy-ds`.

| Script | What it shows |
| --- | --- |
| `henon.jl` | Hénon map as a `Diffeomorphism`; the strange attractor, a bifurcation diagram and the maximum Lyapunov exponent |
| `logistic_3d.jl` | Three coupled logistic maps: fixed-point continuation, period-doubling curve in (alpha, beta) |
| `izhikevich.jl` | Izhikevich neuron as a 1-D Poincaré map with a firing reset; period doubling in (a, I) |
| `duffing.jl` | Forced Duffing oscillator, stroboscopic map; period doublings at B ≈ 0.171687 and 0.368655 |
| `forced_alpazur.jl` | Forced Alpazur oscillator, piecewise-smooth with hysteresis; period doubling in (V, B1) |

## Running

Install the package, a Makie backend and StaticArrays in any environment, then
include a script:

```julia
using Pkg
Pkg.add(["MappyDS", "CairoMakie", "StaticArrays"])
include("examples/duffing.jl")
```

The scripts draw static figures with CairoMakie. The interactive viewer
(`plot2d` without `frames`) needs `using GLMakie` instead; the call is shown
in each script in a non-executed block. Click to restart from a point, `p`
selects a parameter, up/down change it, `f` toggles drawing the trajectories,
space clears the axes.
