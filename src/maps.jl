# Diffeomorphisms built from modes: discrete maps and Poincaré maps.

"""
    AbstractMap

Abstract supertype of maps built from modes ([`PoincareMap`](@ref) and
[`Diffeomorphism`](@ref)). Subtypes implement [`image_detail`](@ref);
[`image`](@ref) and [`trajectory`](@ref) are convenience wrappers.
"""
abstract type AbstractMap end

"""
    MapResult

Image of a point under a map with derivatives (Python: `MapResult`). Fields:
`y`, `jac` (`jac[i, j] = ∂y_i/∂y0_j`), `hes` (`hes[i, j, k] = ∂²y_i/(∂y0_j ∂y0_k)`),
`jac_p`, `hes_p` (parameter derivatives, from MappyDS 0.2 / 0.3),
`param_keys` (keys of the parameter `NamedTuple`), `mode_history` (names of
the modes visited, from the initial to the final one) and `trajectory`.
Derivatives that were not requested are `nothing`.

The characteristic multipliers are available through
`LinearAlgebra.eigvals(res)` and the eigenvectors through
`LinearAlgebra.eigvecs(res)`; see also [`end_mode`](@ref).
"""
struct MapResult
    y::Vector{Float64}
    jac::Union{Nothing, Matrix{Float64}}
    hes::Union{Nothing, Array{Float64, 3}}
    jac_p::Union{Nothing, Matrix{Float64}}
    hes_p::Union{Nothing, Array{Float64, 3}}
    param_keys::Tuple{Vararg{Symbol}}
    mode_history::Vector{Symbol}
    trajectory::Union{Nothing, Trajectory}
end

function MapResult(y; jac = nothing, hes = nothing, jac_p = nothing, hes_p = nothing, param_keys = (), mode_history = Symbol[], trajectory = nothing)
    return MapResult(as_state(y), jac, hes, jac_p, hes_p, Tuple(param_keys), Vector{Symbol}(mode_history), trajectory)
end

"""
    end_mode(res::MapResult) -> Symbol

Name of the mode the image belongs to.
"""
end_mode(res::MapResult) = res.mode_history[end]

function _square_jacobian(res::MapResult)
    res.jac === nothing && throw(ArgumentError("Jacobian was not computed; pass jacobian = true"))
    size(res.jac, 1) == size(res.jac, 2) || throw(ArgumentError("eigenvalues need a square Jacobian (same domain and codomain)"))
    return res.jac
end

LinearAlgebra.eigvals(res::MapResult) = Vector{ComplexF64}(LinearAlgebra.eigvals(_square_jacobian(res)))

LinearAlgebra.eigvecs(res::MapResult) = Matrix{ComplexF64}(LinearAlgebra.eigvecs(_square_jacobian(res)))

_param_keys(::Nothing) = ()
_param_keys(p::NamedTuple) = keys(p)
_param_keys(p::AbstractDict) = Tuple(Symbol(k) for k in keys(p))
_param_keys(p) = ()

# -- chain rule ---------------------------------------------------------------

# Derivatives of `g ∘ f` given those of `f` (so far) and of the step `g`:
# `D(g∘f) = Dg Df`, `D²(g∘f) = D²g[Df, Df] + Dg D²f` and
# `∂_p(g∘f) = Dg ∂_p f + ∂_p g`.
function _compose(jac, hes, jac_p, jac_step, hes_step, jac_p_step)
    jac_step === nothing && return nothing, nothing, nothing
    jac === nothing && return jac_step, hes_step, jac_p_step
    jac_p_new = (jac_p !== nothing && jac_p_step !== nothing) ? jac_step * jac_p + jac_p_step : nothing
    hes_new = nothing
    if hes !== nothing && hes_step !== nothing
        n_out, n_mid = size(jac_step)
        n_in = size(jac, 2)
        hes_new = Array{Float64}(undef, n_out, n_in, n_in)
        for k in 1:n_in, j in 1:n_in, i in 1:n_out
            acc = 0.0
            for l in 1:n_mid
                acc += jac_step[i, l] * hes[l, j, k]
                jlj = jac[l, j]
                for m in 1:n_mid
                    acc += hes_step[i, l, m] * jlj * jac[m, k]
                end
            end
            hes_new[i, j, k] = acc
        end
    end
    return jac_step * jac, hes_new, jac_p_new
end

# -- PoincareMap -----------------------------------------------------------------

"""
    PoincareMap(modes, transitions, initial_mode, end_modes = nothing; max_transitions = 1000)

Poincaré map of a hybrid system described by modes and transition rules.

Starting in `initial_mode`, the state is evolved mode by mode following
`transitions` until one of the `end_modes` (by default `initial_mode`) is
reached; that is one iteration of the map. The Jacobian and Hessian are
composed with the chain rule along the way.

`transitions` maps each mode name (a `Symbol`) to its successor:

* a mode name for a discrete mode, a borderless continuous mode or, as a
  shorthand, a continuous mode with a single border;
* a vector or tuple of mode names, one per border in the order of `borders`;
* a `Dict(:border_name => :mode_name)` covering every border of the mode.

Only modes reachable from `initial_mode` need a rule. The map is validated
at construction: unknown names raise [`UnknownModeError`](@ref); rules that
do not fit their mode, missing rules for reachable modes, dimension
mismatches between a mode and its successor and unreachable end modes raise
[`TransitionError`](@ref). `max_transitions` bounds the number of mode
transitions per iteration.
"""
struct PoincareMap <: AbstractMap
    modes::Dict{Symbol, Mode}
    transitions::Dict{Symbol, Tuple{Vararg{Symbol}}}
    initial_mode::Symbol
    end_modes::Tuple{Vararg{Symbol}}
    max_transitions::Int
end

function PoincareMap(modes, transitions, initial_mode::Symbol, end_modes = nothing; max_transitions::Integer = 1000)
    table = Dict{Symbol, Mode}()
    for m in modes
        m isa Mode || throw(ArgumentError("modes must be ContinuousMode or DiscreteMode, got $(typeof(m))"))
        haskey(table, m.name) && throw(ArgumentError("duplicate mode name $(repr(m.name))"))
        table[m.name] = m
    end
    rules = Dict{Symbol, Tuple{Vararg{Symbol}}}()
    for (src, rule) in pairs(transitions)
        src isa Symbol || throw(ArgumentError("mode names must be symbols, got $(repr(src))"))
        haskey(table, src) || throw(UnknownModeError(src))
        rules[src] = _normalize_rule(table, table[src], rule)
    end
    haskey(table, initial_mode) || throw(UnknownModeError(initial_mode))
    ends = end_modes === nothing ? (initial_mode,) : end_modes isa Symbol ? (end_modes,) : Tuple(end_modes)
    for e in ends
        e isa Symbol || throw(ArgumentError("mode names must be symbols, got $(repr(e))"))
        haskey(table, e) || throw(UnknownModeError(e))
    end
    pm = PoincareMap(table, rules, initial_mode, ends, Int(max_transitions))
    _check_graph(pm)
    return pm
end

# Turn a transition rule into one successor per border.
function _normalize_rule(table::Dict{Symbol, Mode}, mode::Mode, rule)
    names = mode isa ContinuousMode ? border_names(mode) : ()
    n_exits = max(length(names), 1)
    if rule isa Symbol
        n_exits == 1 || throw(TransitionError(mode.name; reason = "the mode has $n_exits borders but the rule names a single successor"))
        targets = (rule,)
    elseif rule isa AbstractDict
        isempty(names) && throw(TransitionError(mode.name; reason = "the mode has no borders to name in a mapping"))
        unknown = sort!([Symbol(k) for k in keys(rule) if !(Symbol(k) in names)])
        absent = sort!([n for n in names if !haskey(rule, n)])
        if !isempty(unknown) || !isempty(absent)
            parts = String[]
            isempty(unknown) || push!(parts, "unknown border(s) $unknown")
            isempty(absent) || push!(parts, "no successor for border(s) $absent")
            throw(TransitionError(mode.name; reason = join(parts, "; ")))
        end
        targets = Tuple(rule[n] for n in names)
    else
        targets = Tuple(rule)
        length(targets) == n_exits || throw(TransitionError(mode.name; reason = "expected $n_exits successor(s) for $(length(names)) border(s), got $(length(targets))"))
    end
    for t in targets
        t isa Symbol || throw(ArgumentError("mode names must be symbols, got $(repr(t))"))
        haskey(table, t) || throw(UnknownModeError(t))
        cod_dim(mode) == dom_dim(table[t]) || throw(TransitionError(mode.name; reason = "it ends in dimension $(cod_dim(mode)) but $(repr(t)) starts in dimension $(dom_dim(table[t]))"))
    end
    return Tuple{Vararg{Symbol}}(targets)
end

# Check that every reachable mode has a rule and that an end mode is reachable.
function _check_graph(pm::PoincareMap)
    seen = Set{Symbol}()
    frontier = [pm.initial_mode]
    reachable_after_start = Set{Symbol}()
    while !isempty(frontier)
        name = pop!(frontier)
        name in seen && continue
        push!(seen, name)
        haskey(pm.transitions, name) || throw(TransitionError(name; reason = "no rule for a reachable mode"))
        for nxt in pm.transitions[name]
            push!(reachable_after_start, nxt)
            nxt in seen || push!(frontier, nxt)
        end
    end
    any(e -> e in reachable_after_start, pm.end_modes) || throw(TransitionError(pm.initial_mode; reason = "no end mode $(collect(pm.end_modes)) is reachable"))
    return nothing
end

function Base.show(io::IO, pm::PoincareMap)
    return print(io, "PoincareMap(", sort!(collect(keys(pm.modes))), ", initial_mode = ", repr(pm.initial_mode), ", end_modes = ", collect(pm.end_modes), ")")
end

dim(pm::PoincareMap) = dom_dim(pm.modes[pm.initial_mode])

function _next_mode(pm::PoincareMap, mode::Mode, border)
    rule = pm.transitions[mode.name]
    index = border === nothing ? 1 : border
    index <= length(rule) || throw(TransitionError(mode.name; border = border))
    return rule[index]
end

function _run(pm::PoincareMap, y::Vector{Float64}, p, start::Symbol, n::Int, jacobian::Bool, hessian::Bool, parameter_jacobian::Bool, trajectory::Bool)
    jac = nothing
    hes = nothing
    jac_p = nothing
    history = Symbol[start]
    traj = Trajectory()
    current = pm.modes[start]
    for _ in 1:n
        reached = false
        for _ in 1:pm.max_transitions
            res = evolve(current, y, p; jacobian, hessian, parameter_jacobian, trajectory)
            y = res.y
            if jacobian
                jac, hes, jac_p = _compose(jac, hes, jac_p, res.jac, res.hes, res.jac_p)
            end
            res.segment === nothing || push!(traj.segments, res.segment)
            nxt = _next_mode(pm, current, res.border)
            push!(history, nxt)
            current = pm.modes[nxt]
            if nxt in pm.end_modes
                reached = true
                break
            end
        end
        reached || error("no end mode reached after $(pm.max_transitions) transitions")
    end
    return y, jac, hes, jac_p, history, traj
end

"""
    image_detail(map, y0, p = nothing; n = 1, jacobian = true, hessian = false,
                 parameter_jacobian = false, trajectory = false, mode = nothing) -> MapResult

Compute the `n`-th image of `y0` under the map with derivatives. `jacobian`
(default `true`) and `hessian` request the Jacobian matrix and the Hessian
tensor; `parameter_jacobian` requests the derivative with respect to every
parameter of the `NamedTuple` `p` (`MapResult.jac_p`, columns in the order
of `MapResult.param_keys`; implies `jacobian`); `trajectory` records the
trajectory. For a [`PoincareMap`](@ref), `mode` starts the iteration in
another mode than `initial_mode` (useful when a previous image ended in a
different end mode).

Requesting derivatives integrates the variational equations together with
the state, which changes the adaptive step sizes of the ODE solver; the
image therefore differs slightly (on the order of the solver tolerance) from
the one obtained with `jacobian = false`.
"""
function image_detail(pm::PoincareMap, y0, p = nothing; n::Integer = 1, jacobian::Bool = true, hessian::Bool = false, parameter_jacobian::Bool = false, trajectory::Bool = false, mode = nothing)
    n >= 1 || throw(ArgumentError("n must be at least 1, got $n"))
    parameter_jacobian && _nparams(p)  # a NamedTuple is required
    start = mode === nothing ? pm.initial_mode : mode
    start isa Symbol || throw(ArgumentError("mode must be a Symbol, got $(repr(start))"))
    haskey(pm.modes, start) || throw(UnknownModeError(start))
    y = as_state(y0)
    length(y) == dom_dim(pm.modes[start]) || throw(ArgumentError("mode $(repr(start)) expects a state of length $(dom_dim(pm.modes[start])), got $(length(y))"))
    y1, jac, hes, jac_p, history, traj = _run(pm, y, p, start, Int(n), jacobian || hessian || parameter_jacobian, hessian, parameter_jacobian, trajectory)
    return MapResult(y1; jac, hes, jac_p, param_keys = _param_keys(p), mode_history = history, trajectory = trajectory ? traj : nothing)
end

"""
    image(map, y0, p = nothing; n = 1, kwargs...) -> Vector{Float64}

Compute the `n`-th image of `y0` under the map.
"""
image(m::AbstractMap, y0, p = nothing; kwargs...) = image_detail(m, y0, p; kwargs..., jacobian = false).y

"""
    trajectory(map, y0, p = nothing; n = 1, kwargs...) -> Trajectory

Compute the trajectory from `y0` over `n` iterations of the map.
"""
function trajectory(m::AbstractMap, y0, p = nothing; kwargs...)
    res = image_detail(m, y0, p; kwargs..., jacobian = false, trajectory = true)
    return res.trajectory
end

# -- Diffeomorphism ---------------------------------------------------------------

"""
    Diffeomorphism(fun::Mapping; name = :map)

A discrete map ``y \\mapsto f(y, p)`` on ``\\mathbb{R}^n`` as a map with a
single mode named `name`; `fun` must have equal domain and codomain
dimensions. The underlying [`Mapping`](@ref) is `d.fun`.
"""
struct Diffeomorphism{F <: Mapping} <: AbstractMap
    fun::F
    mode::DiscreteMode{F}
    pmap::PoincareMap
end

function Diffeomorphism(fun::Mapping; name::Symbol = :map)
    dom_dim(fun) == cod_dim(fun) || throw(ArgumentError("a diffeomorphism must map R^n to R^n"))
    mode = DiscreteMode(name, fun; in_trajectory = true)
    return Diffeomorphism(fun, mode, PoincareMap([mode], Dict(name => name), name))
end

Base.show(io::IO, d::Diffeomorphism) = print(io, "Diffeomorphism(", d.fun.name, ")")

dim(d::Diffeomorphism) = dom_dim(d.fun)

function image_detail(d::Diffeomorphism, y0, p = nothing; n::Integer = 1, jacobian::Bool = true, hessian::Bool = false, parameter_jacobian::Bool = false, trajectory::Bool = false)
    return image_detail(d.pmap, y0, p; n, jacobian, hessian, parameter_jacobian, trajectory)
end
