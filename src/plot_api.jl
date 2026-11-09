# Plotting entry points. The methods live in the package extension
# `MappyDSMakieExt` (ext/MappyDSMakieExt.jl) and become available after a
# Makie backend is loaded (`using GLMakie` for the interactive viewer,
# `using CairoMakie` for static figures). Only the function stubs and the
# configuration types are defined here so that the core package does not
# depend on Makie.

"""
    LineStyle(; color = :black, alpha = 0.3)

Color (any Makie color) and opacity of plotted lines or markers.
"""
Base.@kwdef struct LineStyle
    color::Any = :black
    alpha::Float64 = 0.3
end

"""
    PlotConfig(; kwargs...)

Configuration of [`plot2d`](@ref). Keyword arguments and defaults:

- `only_map = false`: draw only the image points, not the trajectories;
- `figsize = (600, 600)`: figure size in pixels;
- `xlabel = "x"`, `ylabel = "y"`: axis labels;
- `xlim = (-3.0, 3.0)`, `ylim = (-3.0, 3.0)`: axis limits;
- `xkey = 1`, `ykey = 2`: indices of the state components on the axes;
- `linewidth = 1.0`, `markersize = 4.0`: trajectory line width and marker size;
- `param_keys = Symbol[]`, `param_index = 1`, `param_step = 0.1`: parameters
  that can be changed with the up/down keys, the one selected first and the
  increment of one key press;
- `scalar_axis = :x`: for one-dimensional maps, the axis whose coordinate
  becomes the state when the figure is clicked;
- `trajectory_style`, `point_style`, `click_style`: a [`LineStyle`](@ref), or
  for the first two a `Dict{Symbol, LineStyle}` with one style per mode name
  (the key `:_default` is used for modes not listed).
"""
Base.@kwdef struct PlotConfig
    only_map::Bool = false
    figsize::Tuple{Int, Int} = (600, 600)
    xlabel::String = "x"
    ylabel::String = "y"
    xlim::Tuple{Float64, Float64} = (-3.0, 3.0)
    ylim::Tuple{Float64, Float64} = (-3.0, 3.0)
    xkey::Int = 1
    ykey::Int = 2
    linewidth::Float64 = 1.0
    markersize::Float64 = 4.0
    param_keys::Vector{Symbol} = Symbol[]
    param_index::Int = 1
    param_step::Float64 = 0.1
    scalar_axis::Symbol = :x
    trajectory_style::Union{LineStyle, Dict{Symbol, LineStyle}} = LineStyle()
    point_style::Union{LineStyle, Dict{Symbol, LineStyle}} = LineStyle(; color = :red, alpha = 1.0)
    click_style::LineStyle = LineStyle(; color = :blue, alpha = 1.0)
end

const _MAKIE_HINT = "load a Makie backend first (`using GLMakie` for the interactive viewer, `using CairoMakie` for static figures)"

"""
    plot2d(map, y0, p = nothing, config = PlotConfig(); mode = nothing,
           frames = nothing, show = true) -> Figure

Iterate a map and draw its orbit (Python: `mappy_ds.plot.plot2d`). Requires a
Makie backend. Each frame computes one iteration of `map` from the current
state and draws the trajectory segments (unless `config.only_map`) and the
image point.

With `frames = n` the figure is built non-interactively from `n` iterations
and returned (use it with CairoMakie to save a picture). Otherwise the
figure keeps iterating (one iteration per rendered frame, GLMakie) and
reacts to the user:

- mouse click: restart from the clicked coordinates (for a one-dimensional
  state the coordinate on `config.scalar_axis`);
- space or `e`: clear the axis;
- `f`: toggle drawing the trajectories;
- `p`: select the next parameter of `config.param_keys`;
- up/down: change the selected parameter by `config.param_step`.

`mode` starts a [`PoincareMap`](@ref) in another mode than its
`initial_mode`; later iterations continue from the mode each image ends in.
`show = false` skips `display(fig)`.
"""
plot2d(args...; kwargs...) = error("plot2d: $_MAKIE_HINT")

"""
    plot_trajectory(trajectory; ax = nothing, xkey = 1, ykey = 2,
                    styles = LineStyle(; color = :black, alpha = 1.0), linewidth = 1.0) -> Axis

Draw a [`Trajectory`](@ref) onto a Makie `Axis` (a new figure is created if
`ax` is omitted; Python: `mappy_ds.plot.plot_trajectory`). Segments whose
dimension is too small to contain `xkey` and `ykey` are skipped. `styles` is
a [`LineStyle`](@ref) or a `Dict{Symbol, LineStyle}` with one style per mode
name (`:_default` for the others). Requires a Makie backend.
"""
plot_trajectory(args...; kwargs...) = error("plot_trajectory: $_MAKIE_HINT")
