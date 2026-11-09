# Local bifurcations of periodic points and their continuation.

"""
    LocalBfResult

Result of [`find_local_bf`](@ref). Fields: `success`, `retcode`, `y`
(periodic point at the bifurcation), `eigvals`, `eigvecs`, `nfev`,
`residual` (``T^n(y) - y`` followed by the real and imaginary parts of
``\\det(DT^n(y) - e^{iθ} I)`` at the last iterate), `params` (the parameters
with the bifurcation parameter at its solved value) and `theta` (argument
of the critical multiplier). `y`, `eigvals`, `eigvecs`, `params` and `theta`
are `nothing` on failure.
"""
struct LocalBfResult
    success::Bool
    retcode::SciMLBase.ReturnCode.T
    y::Union{Nothing, Vector{Float64}}
    eigvals::Union{Nothing, Vector{ComplexF64}}
    eigvecs::Union{Nothing, Matrix{ComplexF64}}
    nfev::Int
    residual::Vector{Float64}
    params::Union{Nothing, NamedTuple}
    theta::Union{Nothing, Float64}
end

# Cofactor matrix `C` of `a` (so that `d det(a) = Σ C_ij da_ij`), computed from
# minors, which stays well defined where `a` is singular (exactly the
# situation at a bifurcation point, where `C = det(a) a⁻ᵀ` breaks down).
function _cofactor(a::AbstractMatrix)
    d = size(a, 1)
    d == 1 && return ones(eltype(a), 1, 1)
    c = similar(a)
    for j in 1:d, i in 1:d
        rows = [r for r in 1:d if r != i]
        cols = [k for k in 1:d if k != j]
        c[i, j] = (-1)^(i + j) * det(a[rows, cols])
    end
    return c
end

# Residual of the local-bifurcation conditions and its derivatives.
#
# The residual is `(Tⁿ(y) - y, Re det A, Im det A)` with `A = DTⁿ(y) - e^{iθ} I`.
# Returned are the residual, its derivative with respect to `y`
# (`(dim+2) × dim`), with respect to each parameter in `diff_keys`
# (`(dim+2) × length(diff_keys)`) and with respect to `θ` (`dim+2`), all
# analytic from the map's Jacobian, Hessian, parameter derivative and mixed
# derivative, plus the `MapResult`.
function _bifurcation_system(map::AbstractMap, y::Vector{Float64}, p::NamedTuple, theta::Float64, period::Int, diff_keys)
    dim = length(y)
    res = image_detail(map, y, p; n = period, parameter_hessian = true)
    jac = res.jac::Matrix{Float64}
    hes = res.hes::Array{Float64, 3}
    jac_p = res.jac_p::Matrix{Float64}
    hes_p = res.hes_p::Array{Float64, 3}
    mu = cis(theta)
    a = Matrix{ComplexF64}(jac) - mu * I
    dt = det(a)
    cof = _cofactor(a)

    out = zeros(dim + 2)
    out[1:dim] = res.y - y
    out[dim + 1] = real(dt)
    out[dim + 2] = imag(dt)

    d_dy = zeros(dim + 2, dim)
    d_dy[1:dim, :] = jac - I
    for k in 1:dim
        ddet = zero(ComplexF64)  # ∂A_ij/∂y_k = hes[i, j, k]
        for j in 1:dim, i in 1:dim
            ddet += cof[i, j] * hes[i, j, k]
        end
        d_dy[dim + 1, k] = real(ddet)
        d_dy[dim + 2, k] = imag(ddet)
    end

    d_dp = zeros(dim + 2, length(diff_keys))
    for (m, key) in enumerate(diff_keys)
        col = findfirst(==(key), res.param_keys)
        col === nothing && continue  # the map does not depend on it
        ddet = zero(ComplexF64)
        for j in 1:dim, i in 1:dim
            ddet += cof[i, j] * hes_p[i, j, col]
        end
        d_dp[1:dim, m] = jac_p[:, col]
        d_dp[dim + 1, m] = real(ddet)
        d_dp[dim + 2, m] = imag(ddet)
    end

    ddet_dtheta = -im * mu * tr(cof)  # ∂A/∂θ = -i e^{iθ} I
    d_dtheta = zeros(dim + 2)
    d_dtheta[dim + 1] = real(ddet_dtheta)
    d_dtheta[dim + 2] = imag(ddet_dtheta)
    return out, d_dy, d_dp, d_dtheta, res
end

# Residual and square Jacobian for the unknowns `var = (y, λ, θ)`.
function _local_bf_system(map::AbstractMap, var::AbstractVector, params::NamedTuple, param_key::Symbol, period::Int)
    dim = length(var) - 2
    p = merge(params, NamedTuple{(param_key,)}((Float64(var[dim + 1]),)))
    out, d_dy, d_dp, d_dtheta, _ = _bifurcation_system(map, Vector{Float64}(var[1:dim]), p, Float64(var[dim + 2]), period, (param_key,))
    return out, hcat(d_dy, d_dp, d_dtheta)
end

"""
    find_local_bf(map, y0, params::NamedTuple, param_key::Symbol, theta;
                  period = 1, alg = SimpleTrustRegion(), abstol = 1e-10, reltol = 1e-10,
                  maxiters = 100, kwargs...) -> LocalBfResult

Find a local bifurcation point of a periodic point.

A periodic point ``y`` of period ``n`` undergoes a local bifurcation when one
of its characteristic multipliers crosses the unit circle at
``μ = e^{iθ}``. The unknowns ``(y, λ, θ)``, with ``λ`` = `params[param_key]`,
are solved from

```math
T^n(y) - y = 0, \\qquad \\det(DT^n(y) - e^{iθ} I) = 0,
```

the determinant being split into its real and imaginary parts. The initial
`theta` selects the kind of bifurcation: `0` (``μ = 1``) a tangent
(saddle-node) bifurcation, `pi` (``μ = -1``) a period-doubling bifurcation,
any other value a Neimark–Sacker bifurcation (a complex pair ``e^{±iθ}`` on
the unit circle). The Jacobian of the system is supplied analytically from
the map's `jac`, `hes`, `jac_p` and `hes_p`. The nonlinear solver and its
options are as in [`find_cycle`](@ref).
"""
function find_local_bf(
        map::AbstractMap, y0, params::NamedTuple, param_key::Symbol, theta::Real;
        period::Integer = 1, alg = SimpleNonlinearSolve.SimpleTrustRegion(),
        abstol::Real = 1.0e-10, reltol::Real = 1.0e-10, maxiters::Integer = 100, kwargs...,
    )
    period >= 1 || throw(ArgumentError("period must be at least 1, got $period"))
    haskey(params, param_key) || throw(ArgumentError("parameter $(repr(param_key)) is not among the parameters $(keys(params))"))
    y = as_state(y0)
    dim = length(y)
    var0 = vcat(y, Float64(params[param_key]), Float64(theta))
    nfev = Ref(0)
    cache = Ref{Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}}()
    cached = Ref(false)
    evaluate = function (var)
        v = Vector{Float64}(var)
        if cached[] && cache[][1] == v
            return cache[][2], cache[][3]
        end
        nfev[] += 1
        out, jac = _local_bf_system(map, v, params, param_key, Int(period))
        cache[] = (v, out, jac)
        cached[] = true
        return out, jac
    end
    residual = (var, _) -> evaluate(var)[1]
    jacobian = (var, _) -> evaluate(var)[2]
    nf = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian)
    prob = SciMLBase.NonlinearProblem(nf, var0, nothing)
    sol = try
        SciMLBase.solve(prob, alg; abstol, reltol, maxiters, kwargs...)
    catch err
        err isa Union{LinearAlgebra.SingularException, LinearAlgebra.LAPACKException, MappyError, DomainError} || rethrow()
        nothing
    end
    failure(resid) = LocalBfResult(false, SciMLBase.ReturnCode.Failure, nothing, nothing, nothing, nfev[], resid, nothing, nothing)
    sol === nothing && return failure(evaluate(var0)[1])
    u = Vector{Float64}(sol.u)
    resid = evaluate(u)[1]
    ok = SciMLBase.successful_retcode(sol) && all(isfinite, u) && all(isfinite, resid) &&
        maximum(abs, resid) <= max(1000 * abstol, sqrt(abstol))
    ok || return LocalBfResult(false, sol.retcode, nothing, nothing, nothing, nfev[], resid, nothing, nothing)
    y1 = u[1:dim]
    p1 = merge(params, NamedTuple{(param_key,)}((u[dim + 1],)))
    theta1 = u[dim + 2]
    # Same request as inside the system, so that the multipliers are
    # consistent with the system that was solved (the variational equations
    # change the step sequence of the ODE solver).
    detail = image_detail(map, y1, p1; n = period, parameter_hessian = true)
    return LocalBfResult(true, sol.retcode, y1, LinearAlgebra.eigvals(detail), LinearAlgebra.eigvecs(detail), nfev[], resid, p1, theta1)
end

"""
    trace_local_bf(map, y0, params::NamedTuple, bf_param_key::Symbol, theta,
                   cnt_param_key::Symbol, end_value; period = 1, resolution = 100,
                   show_progress = false, callback = nothing, kwargs...) -> Vector{ContinuationStep}

Continue a local bifurcation along a second parameter: `params[cnt_param_key]`
is swept from its current value to `end_value`, and at each point
[`find_local_bf`](@ref) solves for the periodic point, `params[bf_param_key]`
and `θ`, starting from the previous solution. The result traces a
bifurcation curve in the (`cnt_param_key`, `bf_param_key`) plane. Each
step's `y` is the periodic point, `params` holds both parameters and `info`
holds `:theta`, `:eigvals`, `:eigvecs`, `:nfev` and `:residual`. The two
keys must differ.
"""
function trace_local_bf(
        map::AbstractMap, y0, params::NamedTuple, bf_param_key::Symbol, theta::Real, cnt_param_key::Symbol, end_value::Real;
        period::Integer = 1, resolution::Integer = 100, show_progress::Bool = false, callback = nothing, kwargs...,
    )
    bf_param_key == cnt_param_key && throw(ArgumentError("bf_param_key and cnt_param_key must differ"))
    y = as_state(y0)
    dim = length(y)
    step = function (var, p)
        res = find_local_bf(map, var[1:dim], p, bf_param_key, var[dim + 1]; period, kwargs...)
        (res.success && res.y !== nothing && res.theta !== nothing) || return StepFunctionResult(false)
        info = Dict{Symbol, Any}(:theta => res.theta, :eigvals => res.eigvals, :eigvecs => res.eigvecs, :nfev => res.nfev, :residual => res.residual)
        return StepFunctionResult(true, vcat(res.y, res.theta), res.params, info)
    end
    to_public(s::ContinuationStep) = ContinuationStep(s.y[1:dim], s.params, s.info)
    inner_callback = callback === nothing ? nothing : (s -> callback(to_public(s)))
    steps = continuation(step, vcat(y, Float64(theta)), params, cnt_param_key, end_value; resolution, show_progress, callback = inner_callback)
    return ContinuationStep[to_public(s) for s in steps]
end

# Resonances of a Neimark–Sacker curve flagged while `θ` varies: the test
# function changes sign where the critical pair `e^{±iθ}` passes a root of
# unity (1:1 at θ=0, 1:2 at θ=π, 1:3 at 2π/3, 1:4 at π/2).
function _resonance_events(theta_prev::Real, theta::Real)
    events = Symbol[]
    theta_prev == theta && return events
    s0, s1 = sin(theta_prev), sin(theta)
    c0, c1 = cos(theta_prev), cos(theta)
    s0 * s1 < 0 && push!(events, c1 > 0 ? :resonance_1_1 : :resonance_1_2)
    (c0 + 0.5) * (c1 + 0.5) < 0 && push!(events, :resonance_1_3)
    c0 * c1 < 0 && push!(events, :resonance_1_4)
    return events
end

"""
    trace_local_bf_arclength(map, y0, params::NamedTuple, bf_param_key::Symbol, theta,
                             cnt_param_key::Symbol; period = 1, direction = 1, bounds = nothing,
                             step = 0.01, step_min = 1e-6, step_max = 0.1, max_steps = 1000,
                             newton_tol = 1e-10, newton_max_iter = 10,
                             show_progress = false, callback = nothing) -> Vector{ContinuationStep}

Follow a local-bifurcation curve by pseudo-arclength continuation.

The bifurcation conditions of [`find_local_bf`](@ref) are ``d + 2`` equations
in the ``d + 3`` unknowns ``u = (y, λ_{bf}, θ, λ_{cnt})``, so their solutions
form a curve: the bifurcation curve in the ``(λ_{cnt}, λ_{bf})`` plane
together with the periodic point and the argument of the critical
multiplier. The curve is followed with [`arclength_continuation`](@ref), so
it may turn back in ``λ_{cnt}`` (as a fold curve does at a cusp) without
stopping. The Jacobian of the system is fully analytic.

Events between consecutive points are recorded in `info[:events]` of the
later point: `:fold` (the curve turns back in ``λ_{cnt}``; for a fold curve
this happens at a cusp), `:resonance_1_1`, `:resonance_1_2`,
`:resonance_1_3`, `:resonance_1_4` (on a Neimark–Sacker curve, ``θ`` passes
``0``, ``π``, ``2π/3`` or ``π/2``) and `:second_multiplier_crossing` (the
number of multipliers outside the unit circle changes, i.e. another
bifurcation curve is crossed). Each step's `info` also holds `:theta`,
`:eigvals`, `:eigvecs`, `:tangent` (unit vector in the ``u`` space), `:step`
and `:newton_iterations`. `direction` and `bounds` refer to ``λ_{cnt}``.

# Examples

The cusp normal form `x ↦ x + l1 + l2 x - x³` has the fold curve
`l2 = 3x²`, `l1 = -2x³`, which turns back in `l2` at the cusp `(0, 0)`:

```julia
cusp = Mapping((x, p) -> x[1] + p.l1 + p.l2 * x[1] - x[1]^3, 1)
steps = trace_local_bf_arclength(Diffeomorphism(cusp), [0.5], (l1 = -0.25, l2 = 0.75), :l1, 0.0, :l2;
    direction = -1, step = 0.1, step_max = 0.2, bounds = (-1.0, 0.75))
any(:fold in s.info[:events] for s in steps)           # true
steps[end].y[1] < 0 && steps[end].params.l1 > 0        # true: the lower branch
```
"""
function trace_local_bf_arclength(
        map::AbstractMap, y0, params::NamedTuple, bf_param_key::Symbol, theta::Real, cnt_param_key::Symbol;
        period::Integer = 1, direction::Integer = 1, bounds = nothing,
        step::Real = 0.01, step_min::Real = 1.0e-6, step_max::Real = 0.1, max_steps::Integer = 1000,
        newton_tol::Real = 1.0e-10, newton_max_iter::Integer = 10,
        show_progress::Bool = false, callback = nothing,
    )
    bf_param_key == cnt_param_key && throw(ArgumentError("bf_param_key and cnt_param_key must differ"))
    for key in (bf_param_key, cnt_param_key)
        haskey(params, key) || throw(ArgumentError("parameter $(repr(key)) is not among the parameters $(keys(params))"))
    end
    keys_ = (bf_param_key, cnt_param_key)
    y = as_state(y0)
    dim = length(y)
    last = Ref{Union{Nothing, MapResult}}(nothing)
    with_params(bf, cnt) = merge(params, NamedTuple{keys_}((Float64(bf), Float64(cnt))))

    system = function (u)
        p = with_params(u[dim + 1], u[dim + 3])
        out, d_dy, d_dp, d_dtheta, res = _bifurcation_system(map, u[1:dim], p, Float64(u[dim + 2]), Int(period), keys_)
        last[] = res
        # columns: y, λ_bf, θ, λ_cnt
        return out, hcat(d_dy, d_dp[:, 1], d_dtheta, d_dp[:, 2])
    end

    steps = ContinuationStep[]
    prev_tangent = Ref(Float64[])
    prev_theta = Ref(0.0)
    prev_outside = Ref(0)
    on_point = function (point::ArclengthPoint)
        res = last[]
        res === nothing && error("the continuation accepted a point before evaluating the map")
        eig = LinearAlgebra.eigvals(res)
        eigvec = LinearAlgebra.eigvecs(res)
        th = point.u[dim + 2]
        outside = count(l -> abs(l) > 1.0 + 1.0e-6, eig)
        events = Symbol[]
        if !isempty(steps)
            prev_tangent[][end] * point.tangent[end] < 0 && push!(events, :fold)
            append!(events, _resonance_events(prev_theta[], th))
            outside != prev_outside[] && push!(events, :second_multiplier_crossing)
        end
        prev_tangent[] = point.tangent
        prev_theta[] = th
        prev_outside[] = outside
        info = Dict{Symbol, Any}(
            :theta => th, :eigvals => eig, :eigvecs => eigvec, :tangent => point.tangent,
            :step => point.step, :newton_iterations => point.newton_iterations, :events => events,
        )
        cs = ContinuationStep(point.u[1:dim], with_params(point.u[dim + 1], point.u[dim + 3]), info)
        push!(steps, cs)
        callback === nothing || callback(cs)
        if show_progress
            tag = isempty(events) ? "" : " " * join(string.(events), ",")
            print(
                stderr, "ARCLENGTH ", lpad(length(steps), 5),
                " ", cnt_param_key, "=", Printf.@sprintf("%+.10g", cs.params[cnt_param_key]),
                " ", bf_param_key, "=", Printf.@sprintf("%+.10g", cs.params[bf_param_key]),
                " θ=", Printf.@sprintf("%+.6f", th), " h=", Printf.@sprintf("%.3g", point.step), tag, "\r",
            )
            flush(stderr)
        end
        return nothing
    end

    u0 = vcat(y, Float64(params[bf_param_key]), Float64(theta), Float64(params[cnt_param_key]))
    try
        arclength_continuation(
            system, u0;
            step, step_min, step_max, max_steps, direction, bounds, newton_tol, newton_max_iter, callback = on_point,
        )
    finally
        show_progress && println(stderr)
    end
    return steps
end
