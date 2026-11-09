# Continuous and discrete modes of a hybrid dynamical system.

"""
    StepResult

Result of [`evolve`](@ref). Fields: `y` (state at the end of the mode),
`jac` (`jac[i, j] = ∂y_i/∂y0_j`), `hes` (`hes[i, j, k] = ∂²y_i/(∂y0_j ∂y0_k)`),
`jac_p` and `hes_p` (parameter derivatives, from MappyDS 0.2 / 0.3),
`border` (1-based index of the border that terminated a continuous mode,
`nothing` otherwise), `t` (time spent in a continuous mode) and `segment`
(the recorded [`TrajectorySegment`](@ref)). Derivatives that were not
requested are `nothing`.
"""
struct StepResult
    y::Vector{Float64}
    jac::Union{Nothing, Matrix{Float64}}
    hes::Union{Nothing, Array{Float64, 3}}
    jac_p::Union{Nothing, Matrix{Float64}}
    hes_p::Union{Nothing, Array{Float64, 3}}
    border::Union{Nothing, Int}
    t::Union{Nothing, Float64}
    segment::Union{Nothing, TrajectorySegment}
end

function StepResult(y; jac = nothing, hes = nothing, jac_p = nothing, hes_p = nothing, border = nothing, t = nothing, segment = nothing)
    return StepResult(y, jac, hes, jac_p, hes_p, border, t, segment)
end

"""
    Mode

Abstract supertype of [`ContinuousMode`](@ref) and [`DiscreteMode`](@ref): a
primitive, smooth piece of a hybrid system together with the rule that
terminates it. Both kinds expose [`evolve`](@ref).
"""
abstract type Mode end

dom_dim(m::Mode) = dom_dim(m.fun)
cod_dim(m::Mode) = cod_dim(m.fun)
Base.nameof(m::Mode) = m.name

"""
    DiscreteMode(name, fun::Mapping; in_trajectory = false)

A mode whose evolution is a single application of the map `fun`.
`in_trajectory` decides whether the jump contributes a two-point
[`TrajectorySegment`](@ref) to recorded trajectories.
"""
struct DiscreteMode{F <: Mapping} <: Mode
    name::Symbol
    fun::F
    in_trajectory::Bool
end

DiscreteMode(name::Symbol, fun::Mapping; in_trajectory::Bool = false) = DiscreteMode(name, fun, in_trajectory)

Base.show(io::IO, m::DiscreteMode) = print(io, "DiscreteMode(", repr(m.name), ", ", m.fun.name, ")")

"""
    ContinuousMode(name, fun::VectorField; borders = (), max_interval = 20.0,
                   in_trajectory = true, trajectory_dt = 1.0e-2,
                   alg = Tsit5(), reltol = 1.0e-6, abstol = 1.0e-9, solver_kwargs...)

A mode whose evolution follows the autonomous ODE `fun` until one of the
`borders` ([`Border`](@ref)s of the same dimension, with distinct names) is
crossed in the direction it declares. Without borders the mode is the
time-`max_interval` flow map. `alg` is any OrdinaryDiffEq algorithm (the
default `Tsit5()` corresponds to SciPy's `RK45`); `reltol`, `abstol` and the
remaining keyword arguments are forwarded to `solve`.

When derivatives are requested the first and second variational equations
are integrated together with the state and the result is corrected for the
dependence of the hitting time on the initial state (Miino, NOLTA, IEICE
15(2), 2024). The image then differs from the one obtained without
derivatives by an amount of the order of the solver tolerance, because the
adaptive step sizes change.
"""
struct ContinuousMode{F <: VectorField, B <: Tuple, A, K <: NamedTuple} <: Mode
    name::Symbol
    fun::F
    borders::B
    max_interval::Float64
    in_trajectory::Bool
    trajectory_dt::Float64
    alg::A
    reltol::Float64
    abstol::Float64
    solver_kwargs::K
end

function ContinuousMode(
        name::Symbol, fun::VectorField;
        borders = (), max_interval::Real = 20.0, in_trajectory::Bool = true,
        trajectory_dt::Real = 1.0e-2, alg = OrdinaryDiffEqTsit5.Tsit5(),
        reltol::Real = 1.0e-6, abstol::Real = 1.0e-9, solver_kwargs...,
    )
    bs = borders isa Border ? (borders,) : Tuple(borders)
    for b in bs
        b isa Border || throw(ArgumentError("borders must be declared with Border(...), got $(typeof(b))"))
        dom_dim(b) == dom_dim(fun) || throw(ArgumentError("border $(repr(b.name)) has dim $(dom_dim(b)), expected $(dom_dim(fun))"))
    end
    names = map(b -> b.name, bs)
    length(unique(names)) == length(names) || throw(ArgumentError("borders of mode $(repr(name)) must have distinct names; pass name = ... to Border(...)"))
    return ContinuousMode(
        name, fun, bs, Float64(max_interval), in_trajectory, Float64(trajectory_dt),
        alg, Float64(reltol), Float64(abstol), (; solver_kwargs...),
    )
end

Base.show(io::IO, m::ContinuousMode) = print(io, "ContinuousMode(", repr(m.name), ", ", m.fun.name, ")")

"""
    border_names(mode::ContinuousMode) -> Tuple

Names of the borders of the mode, in the order of `borders`.
"""
border_names(m::ContinuousMode) = map(b -> b.name, m.borders)

# -- evolve -------------------------------------------------------------------

"""
    evolve(mode, y0, p = nothing; jacobian = false, hessian = false,
           parameter_jacobian = false, trajectory = false) -> StepResult

Evolve the state `y0` through the mode (Python: `Mode.step`). `jacobian`
and `hessian` request the Jacobian matrix and the Hessian tensor of the mode
map (`hessian` implies `jacobian`); `parameter_jacobian` requests the
derivative with respect to the parameters (`p` must then be a `NamedTuple`;
one column per key, in order); `trajectory` records the trajectory within
the mode if the mode is `in_trajectory`.
"""
function evolve end

function evolve(mode::DiscreteMode, y0, p = nothing; jacobian::Bool = false, hessian::Bool = false, parameter_jacobian::Bool = false, trajectory::Bool = false)
    fun = mode.fun
    y = _svec(fun, y0)
    y1 = _eval(fun, y, p)
    jac = (jacobian || hessian) ? Matrix(_jac(fun, y, p)) : nothing
    hes = hessian ? Array(_hes(fun, y, p)) : nothing
    jac_p = parameter_jacobian ? Matrix(_jac_p(fun, y, p)) : nothing
    segment = nothing
    if trajectory && mode.in_trajectory && dom_dim(fun) == cod_dim(fun)
        segment = TrajectorySegment(mode.name, :discrete, hcat(Vector(y), Vector(y1)))
    end
    return StepResult(Vector(y1); jac, hes, jac_p, segment)
end

function evolve(mode::ContinuousMode{<:VectorField{N}}, y0, p = nothing; jacobian::Bool = false, hessian::Bool = false, parameter_jacobian::Bool = false, trajectory::Bool = false) where {N}
    y = as_state(y0)
    length(y) == N || throw(ArgumentError("expected a state of length $N, got $(length(y))"))
    jacobian = jacobian || hessian || parameter_jacobian
    K = parameter_jacobian ? _nparams(p) : 0
    z0 = copy(y)
    jacobian && append!(z0, vec(Matrix{Float64}(I, N, N)))
    parameter_jacobian && append!(z0, zeros(N * K))
    hessian && append!(z0, zeros(N^3))

    fired = Ref(0)
    callbacks = _callbacks(mode, p, fired)
    record = trajectory && mode.in_trajectory
    prob = SciMLBase.ODEProblem(_rhs(mode, p, jacobian, hessian, parameter_jacobian, K), z0, (0.0, mode.max_interval))
    sol = SciMLBase.solve(
        prob, mode.alg;
        reltol = mode.reltol, abstol = mode.abstol, callback = callbacks,
        save_everystep = record, dense = record, save_start = record,
        mode.solver_kwargs...,
    )

    rc = sol.retcode
    border = nothing
    if rc == SciMLBase.ReturnCode.Terminated
        border = fired[]
        border >= 1 || error("solve of mode $(repr(mode.name)) terminated without a border event")
    elseif rc == SciMLBase.ReturnCode.Success
        isempty(mode.borders) || throw(ModeNotTerminatedError(mode.name, mode.max_interval))
    else
        error("solve failed in mode $(repr(mode.name)): $rc")
    end

    z1 = sol.u[end]
    t1 = Float64(sol.t[end])
    y1 = z1[1:N]
    jac = nothing
    hes = nothing
    jac_p = nothing
    if jacobian
        off = N
        jac = reshape(z1[(off + 1):(off + N * N)], N, N)
        off += N * N
        if parameter_jacobian
            jac_p = reshape(z1[(off + 1):(off + N * K)], N, K)
            off += N * K
        end
        if hessian
            hes = reshape(z1[(off + 1):(off + N^3)], N, N, N)
        end
        if border !== nothing
            jac, hes, jac_p = _border_correction(mode, _head(Val(N), y1), p, mode.borders[border], jac, hes, jac_p)
        end
    end

    segment = nothing
    if record
        n_pts = max(ceil(Int, t1 / mode.trajectory_dt), 1) + 1
        ts = collect(range(0.0, t1; length = n_pts))
        Y = Matrix{Float64}(undef, N, n_pts)
        for (k, t) in enumerate(ts)
            u = sol(t)
            for i in 1:N
                Y[i, k] = u[i]
            end
        end
        # The dense output of the last (truncated) step is not exact at the
        # event time; the segment ends at the state the mode actually ends in.
        Y[:, end] = y1
        segment = TrajectorySegment(mode.name, :continuous, Y, ts)
    end

    return StepResult(y1; jac, hes, jac_p, border, t = t1, segment)
end

# Right-hand side of the state augmented with the variational systems. Layout
# of the augmented state: y (N), then J (N²) if `jacobian`, then P (N·K) if
# `parameter_jacobian`, then H (N³) if `hessian`:
#
#   J' = Df J,   P' = Df P + ∂f/∂p,   H'_{ijk} = Σ (D²f)_{ilm} J_{lj} J_{mk} + Σ (Df)_{il} H_{ljk}.
function _rhs(mode::ContinuousMode{<:VectorField{N}}, p, jacobian::Bool, hessian::Bool, parameter_jacobian::Bool, K::Int) where {N}
    fun = mode.fun
    return function (dz, z, _, t)
        y = _head(Val(N), z)
        fy = _eval(fun, y, p)
        @inbounds for i in 1:N
            dz[i] = fy[i]
        end
        if jacobian
            Df = _jac(fun, y, p)
            J = reshape(view(z, (N + 1):(N + N * N)), N, N)
            dJ = reshape(view(dz, (N + 1):(N + N * N)), N, N)
            @inbounds for j in 1:N, i in 1:N
                s = 0.0
                for l in 1:N
                    s += Df[i, l] * J[l, j]
                end
                dJ[i, j] = s
            end
            off = N + N * N
            if parameter_jacobian
                fp = _jac_p(fun, y, p)
                P = reshape(view(z, (off + 1):(off + N * K)), N, K)
                dP = reshape(view(dz, (off + 1):(off + N * K)), N, K)
                @inbounds for m in 1:K, i in 1:N
                    s = fp[i, m]
                    for l in 1:N
                        s += Df[i, l] * P[l, m]
                    end
                    dP[i, m] = s
                end
                off += N * K
            end
            if hessian
                D2f = _hes(fun, y, p)
                H = reshape(view(z, (off + 1):(off + N^3)), N, N, N)
                dH = reshape(view(dz, (off + 1):(off + N^3)), N, N, N)
                @inbounds for k in 1:N, j in 1:N, i in 1:N
                    s = 0.0
                    for l in 1:N
                        s += Df[i, l] * H[l, j, k]
                        Jlj = J[l, j]
                        for m in 1:N
                            s += D2f[i, l, m] * Jlj * J[m, k]
                        end
                    end
                    dH[i, j, k] = s
                end
            end
        end
        return nothing
    end
end

# One ContinuousCallback per border; the index of the border that fired is
# stored in `fired`. The direction of a border selects which crossing calls
# `terminate!`.
function _callbacks(mode::ContinuousMode{<:VectorField{N}}, p, fired::Base.RefValue{Int}) where {N}
    isempty(mode.borders) && return nothing
    cbs = map(eachindex(mode.borders)) do i
        b = mode.borders[i]
        condition = (u, t, integrator) -> _eval(b, _head(Val(N), u), p)[1]
        stop = function (integrator)
            fired[] = i
            SciMLBase.terminate!(integrator)
            return nothing
        end
        positive = b.direction >= 0 ? stop : nothing
        negative = b.direction <= 0 ? stop : nothing
        return SciMLBase.ContinuousCallback(condition, positive, negative; save_positions = (false, false))
    end
    return SciMLBase.CallbackSet(cbs...)
end

# Account for the dependence of the hitting time on the initial state.
#
# With `f` the vector field, `q` the border function and `J`, `H` the
# derivatives of the flow at the hitting time `τ`, the mode map
# `y0 ↦ φ(τ(y0), y0)` has Jacobian `B J` with `B = I - f ⊗ ∇q / (∇q · f)`.
# The Hessian follows by differentiating once more with respect to `y0`,
# taking `∂τ/∂y0 = -(∇q J) / (∇q · f)` into account; every quantity evaluated
# at the hitting point depends on `y0` through the mode Jacobian `B J`.
#
# The parameter derivative of the mode map is `B P + G` with
# `G = -f ⊗ ∂q/∂p / (∇q · f)` (the border may depend on the parameters).
function _border_correction(mode::ContinuousMode, y1::SVector{N}, p, b::Border, Jflow::Matrix{Float64}, Hflow::Union{Nothing, Array{Float64, 3}}, Pflow::Union{Nothing, Matrix{Float64}}) where {N}
    fun = mode.fun
    f = _eval(fun, y1, p)
    q = _jac(b, y1, p)[1, :]
    s = dot(q, f)
    s == 0.0 && throw(DomainError(s, "border $(repr(b.name)) of mode $(repr(mode.name)) is tangent to the flow"))
    B = Matrix{Float64}(I, N, N)
    for d in 1:N, a in 1:N
        B[a, d] -= f[a] * q[d] / s
    end
    jac = B * Jflow
    jac_p = nothing
    if Pflow !== nothing
        qp = _jac_p(b, y1, p)
        jac_p = B * Pflow
        for m in 1:size(jac_p, 2), a in 1:N
            jac_p[a, m] -= f[a] * qp[1, m] / s
        end
    end
    Hflow === nothing && return jac, nothing, jac_p

    Df = _jac(fun, y1, p)
    Hq = _hes(b, y1, p)
    ds = zeros(N)
    for e in 1:N
        acc = 0.0
        for a in 1:N
            acc += Hq[1, a, e] * f[a] + q[a] * Df[a, e]
        end
        ds[e] = acc
    end
    dB = Array{Float64}(undef, N, N, N)
    for e in 1:N, d in 1:N, a in 1:N
        dB[a, d, e] = -(Df[a, e] * q[d] + f[a] * Hq[1, d, e]) / s + f[a] * q[d] * ds[e] / s^2
    end
    tau = zeros(N)
    for c in 1:N
        acc = 0.0
        for a in 1:N
            acc += q[a] * Jflow[a, c]
        end
        tau[c] = -acc / s
    end
    DfJ = Df * Jflow
    hes = Array{Float64}(undef, N, N, N)
    for c in 1:N, bb in 1:N, a in 1:N
        acc = 0.0
        for d in 1:N
            inner = 0.0
            for e in 1:N
                inner += dB[a, d, e] * jac[e, c]
            end
            acc += inner * Jflow[d, bb] + B[a, d] * (Hflow[d, bb, c] + DfJ[d, bb] * tau[c])
        end
        hes[a, bb, c] = acc
    end
    return jac, hes, jac_p
end
