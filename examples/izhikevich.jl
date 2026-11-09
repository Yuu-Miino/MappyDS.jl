# # Izhikevich neuron
#
# A hybrid system with a firing reset. The membrane potential ``v`` and the
# recovery variable ``u`` follow
#
# ```math
# \dot v = 0.04 v^2 + 5 v + 140 - u + I, \qquad \dot u = a (b v - u)
# ```
#
# until ``v`` reaches 30 (firing), when ``(v, u)`` jumps to ``(c, u + d)``.
# Sampling ``u`` right after each firing gives a one-dimensional Poincaré map
# built from four modes (Miino 2024):
#
# * `m0` (1 → 2): lift ``u`` to the reset point ``(c, u)``;
# * `m1` (2 → 2): integrate the ODE until the border ``v = 30``;
# * `m2` (2 → 2): firing jump ``(v, u) \to (v - 30 + c, u + d)``;
# * `m3` (2 → 1): keep ``u``.
#
# Python counterpart: `examples/izhikevich.py` of `mappy-ds`.

using MappyDS, CairoMakie, StaticArrays

izhikevich = VectorField(2; name = :izhikevich) do y, p
    v, u = y
    return SVector(0.04 * v^2 + 5 * v + 140 - u + p.I, p.a * (p.b * v - u))
end
fire_border = Border((y, _) -> y[1] - 30, 2; direction = 1, name = :fire)
jump = Mapping((y, p) -> y + SVector(p.c - 30, p.d), 2; name = :jump)
lift = Mapping((u, p) -> SVector(p.c, u[1]), 1, 2; name = :lift)
keep_u = Mapping((y, _) -> y[2], 2, 1; name = :keep_u)

# The bifurcation analysis below solves systems to a tolerance of `1e-10`, so
# the ODE is integrated more accurately than with the default tolerances.

pmap = PoincareMap(
    [
        DiscreteMode(:m0, lift),
        ContinuousMode(:m1, izhikevich; borders = (fire_border,), reltol = 1.0e-8, abstol = 1.0e-10),
        DiscreteMode(:m2, jump; in_trajectory = true),
        DiscreteMode(:m3, keep_u),
    ],
    Dict(:m0 => :m1, :m1 => [:m2], :m2 => :m3, :m3 => :m0),
    :m0,
)

params = (a = 0.2, b = 0.2, c = -50.0, d = 2.0, I = 10.0)
u0 = -3.0

# ## Orbit
#
# [`trajectory`](@ref) records the states visited during `n` iterations of
# the map; [`plot_trajectory`](@ref) draws the continuous flow (`m1`, teal)
# and the firing jumps (`m2`, orange; they are recorded because `m2` was
# declared with `in_trajectory = true`).

styles = Dict(:m1 => LineStyle(; color = :teal, alpha = 1.0), :m2 => LineStyle(; color = :orange, alpha = 1.0))
traj = trajectory(pmap, u0, params; n = 30)
fig = Figure(; size = (600, 400))
ax = Axis(fig[1, 1]; xlabel = "v", ylabel = "u", limits = (-80, 35, -10, 2))
plot_trajectory(traj; ax, styles)
fig

# The interactive viewer uses the same styles. Since the map is
# one-dimensional, a click restarts from the clicked `u` (the vertical
# coordinate, `scalar_axis = :y`); `p` switches between `a` and `I`.
#
# ```julia
# using GLMakie
# config = PlotConfig(;
#     xlim = (-80.0, 35.0), ylim = (-10.0, 2.0), xlabel = "v", ylabel = "u",
#     param_keys = [:a, :I], scalar_axis = :y, trajectory_style = styles,
# )
# plot2d(pmap, u0, params, config)
# ```

# ## Fixed point and its continuation
#
# One iteration of the map, its derivative and the modes visited:

res = image_detail(pmap, u0, params)
res.y, res.jac, res.mode_history

# [`find_cycle`](@ref) finds the fixed point (a periodic firing pattern) and
# [`trace_cycle`](@ref) follows it while ``I`` decreases to 4.5:

cycle = find_cycle(pmap, u0, params)
steps = trace_cycle(pmap, cycle.y, params, :I, 4.5; resolution = 20)
println("fixed point at I = $(params.I): u = $(cycle.y[1])")
for s in steps
    println("I = $(round(s.params.I; digits = 4))  u = $(round(s.y[1]; digits = 6))  multiplier = $(round(real(s.info[:eigvals][1]); digits = 6))")
end

# ## Period doubling
#
# The multiplier crosses ``-1`` along the branch. Starting from the point
# where it is closest to ``-1``, [`find_local_bf`](@ref) locates the
# period-doubling bifurcation (``\theta = \pi``), and [`trace_local_bf`](@ref)
# follows it in the ``(a, I)`` plane while ``a`` increases to 0.3.

multipliers = [real(s.info[:eigvals][1]) for s in steps]
k = argmin(abs.(multipliers .+ 1))
bf = find_local_bf(pmap, steps[k].y, steps[k].params, :I, pi)
println("period doubling at I = $(bf.params.I), u = $(bf.y[1])")

curve = trace_local_bf(pmap, bf.y, bf.params, :I, bf.theta, :a, 0.3; resolution = 10)
println("period-doubling curve ($(length(curve)) points):")
println("       a         I")
for s in curve
    println("  $(round(s.params.a; digits = 5))  $(round(s.params.I; digits = 5))")
end

fig = Figure(; size = (600, 350))
ax = Axis(fig[1, 1]; xlabel = "a", ylabel = "I", title = "period-doubling curve")
lines!(ax, [s.params.a for s in curve], [s.params.I for s in curve]; color = :red)
scatter!(ax, [s.params.a for s in curve], [s.params.I for s in curve]; color = :red)
fig
