# Plotting

Plotting lives in a package extension that is loaded together with a Makie
backend: `using GLMakie` for the interactive viewer, `using CairoMakie` for
static figures (the core package does not depend on Makie).

!!! note "Name clash with `image`"
    Makie also exports a function named `image` (the image plot). With both
    packages loaded, `image` has to be qualified or imported explicitly:
    `using MappyDS: image` (or call `MappyDS.image`).

## Trajectories

[`plot_trajectory`](@ref) draws a [`Trajectory`](@ref) segment by segment
onto a Makie `Axis`, with one [`LineStyle`](@ref) for all modes or a
`Dict{Symbol, LineStyle}` with one style per mode name:

```julia
using MappyDS, CairoMakie

traj = trajectory(pmap, -3.0, params; n = 5)
ax = plot_trajectory(traj; styles = Dict(:A => LineStyle(; color = :teal, alpha = 0.8)))
save("izhikevich.png", ax.parent)
```

## Iterating a map interactively

[`plot2d`](@ref) iterates a map and draws its orbit: each frame computes one
iteration from the current state and draws the trajectory segments and the
image point. With GLMakie the figure keeps iterating and reacts to the
user (click to restart from the clicked point, `f` to toggle the
trajectories, space to clear, `p` to select a parameter and up/down to
change it):

```julia
using MappyDS, GLMakie

config = PlotConfig(;
    xlim = (-60.0, 40.0), ylim = (-10.0, 5.0), xlabel = "v", ylabel = "u",
    param_keys = [:a, :I], param_step = 0.1,
    trajectory_style = Dict(:A => LineStyle(; color = :teal, alpha = 0.5)),
)
plot2d(pmap, -3.0, params, config)
```

With `frames = n` the figure is built from `n` iterations and returned,
which is the way to produce a picture with CairoMakie:

```julia
using CairoMakie
fig = plot2d(henon_map, [0.1, 0.1], (a = 1.4, b = 0.3), PlotConfig(; xlim = (-2.0, 2.0), ylim = (-0.5, 0.5), only_map = true); frames = 2000, show = false)
save("henon.png", fig)
```
