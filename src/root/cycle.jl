# Periodic points (cycles) of maps and their continuation.

"""
    CycleResult

Result of [`find_cycle`](@ref). Fields: `success`, `retcode`
(`SciMLBase.ReturnCode` of the nonlinear solver), `y` (the periodic point,
`nothing` on failure), `eigvals` and `eigvecs` (characteristic multipliers
and eigenvectors of the Jacobian of the `period`-th iterate at `y`,
`nothing` on failure), `nfev` (number of evaluations of the map) and
`residual` (`Tⁿ(y) - y` at the last iterate).
"""
struct CycleResult
    success::Bool
    retcode::SciMLBase.ReturnCode.T
    y::Union{Nothing, Vector{Float64}}
    eigvals::Union{Nothing, Vector{ComplexF64}}
    eigvecs::Union{Nothing, Matrix{ComplexF64}}
    nfev::Int
    residual::Vector{Float64}
end

"""
    find_cycle(map, y0, p = nothing; period = 1, alg = SimpleTrustRegion(),
               abstol = 1.0e-10, reltol = 1.0e-10, maxiters = 100, kwargs...) -> CycleResult

Find a periodic point of a map by solving ``T^n(y) - y = 0`` with the
analytic Jacobian ``DT^n(y) - I`` provided by the map, where ``T`` is `map`
and ``n`` is `period`. The system is a `SciMLBase.NonlinearProblem`; `alg`
is any algorithm of SimpleNonlinearSolve or NonlinearSolve, and the
remaining keyword arguments are forwarded to `solve`.

# Examples

```julia
henon = Mapping(2) do y, p
    x, u = y
    return SVector(1 - p.a * x^2 + u, p.b * x)
end
res = find_cycle(Diffeomorphism(henon), [0.6, 0.2], (a = 1.4, b = 0.3))
res.y        # ≈ [0.631354, 0.189406]
res.eigvals  # ≈ [-1.923739, 0.155946]
```
"""
function find_cycle(map::AbstractMap, y0, p = nothing; period::Integer = 1, alg = SimpleNonlinearSolve.SimpleTrustRegion(), abstol::Real = 1.0e-10, reltol::Real = 1.0e-10, maxiters::Integer = 100, kwargs...)
    period >= 1 || throw(ArgumentError("period must be at least 1, got $period"))
    y = as_state(y0)
    nfev = Ref(0)
    residual = function (u, _)
        nfev[] += 1
        return image(map, u, p; n = period) - u
    end
    jac = function (u, _)
        res = image_detail(map, u, p; n = period, jacobian = true)
        return res.jac - I
    end
    nf = SciMLBase.NonlinearFunction{false}(residual; jac = jac)
    prob = SciMLBase.NonlinearProblem(nf, y, nothing)
    sol = try
        SciMLBase.solve(prob, alg; abstol, reltol, maxiters, kwargs...)
    catch err
        # A singular Jacobian (e.g. a search that wanders to a critical point
        # of a map without fixed points) or a mode that cannot be evaluated
        # at an iterate is a failed search, not a programming error.
        err isa Union{LinearAlgebra.SingularException, LinearAlgebra.LAPACKException, MappyError, DomainError} || rethrow()
        nothing
    end
    if sol === nothing
        return CycleResult(false, SciMLBase.ReturnCode.Failure, nothing, nothing, nothing, nfev[], residual(y, nothing))
    end
    u = Vector{Float64}(sol.u)
    resid = residual(u, nothing)
    nfev[] -= 1
    # `Stalled` counts as successful for nonlinear solvers; require a small
    # residual as well (ODE-based maps have a noise floor of ~1.0e-9 at the
    # default solver tolerances).
    ok = SciMLBase.successful_retcode(sol) && all(isfinite, u) && all(isfinite, resid) &&
        maximum(abs, resid) <= max(1000 * abstol, sqrt(abstol))
    if ok
        detail = image_detail(map, u, p; n = period, jacobian = true)
        return CycleResult(true, sol.retcode, u, LinearAlgebra.eigvals(detail), LinearAlgebra.eigvecs(detail), nfev[], resid)
    end
    return CycleResult(false, sol.retcode, nothing, nothing, nothing, nfev[], resid)
end

"""
    trace_cycle(map, y0, params::NamedTuple, param_key::Symbol, end_value;
                period = 1, resolution = 100, show_progress = false, callback = nothing,
                kwargs...) -> Vector{ContinuationStep}

Continue a periodic point along a parameter: calls [`find_cycle`](@ref) at
each point of a [`continuation`](@ref) of `params[param_key]` from its
current value to `end_value`, starting each search from the previous
periodic point. The `info` of each step holds `:eigvals`, `:eigvecs`,
`:nfev` and `:residual` as in [`CycleResult`](@ref). The remaining keyword
arguments are forwarded to `find_cycle`.
"""
function trace_cycle(map::AbstractMap, y0, params::NamedTuple, param_key::Symbol, end_value::Real; period::Integer = 1, resolution::Integer = 100, show_progress::Bool = false, callback = nothing, kwargs...)
    step = function (y, p)
        res = find_cycle(map, y, p; period, kwargs...)
        info = Dict{Symbol, Any}(:eigvals => res.eigvals, :eigvecs => res.eigvecs, :nfev => res.nfev, :residual => res.residual)
        return StepFunctionResult(res.success, res.y, nothing, info)
    end
    return continuation(step, as_state(y0), params, param_key, end_value; resolution, show_progress, callback)
end

# -- pseudo-arclength continuation of periodic points ---------------------------------

# Bifurcations between two consecutive points, from sign changes.
function _classify_events(prev_tangent, tangent, prev_jac, jac, prev_eig, eig)
    events = Symbol[]
    fold_turn = prev_tangent[end] * tangent[end] < 0
    det_minus = det(prev_jac - I) * det(jac - I) < 0
    if fold_turn
        push!(events, :fold)
    elseif det_minus
        push!(events, :branch_point)
    end
    det(prev_jac + I) * det(jac + I) < 0 && push!(events, :period_doubling)
    # A complex pair crossing the unit circle changes the number of
    # multipliers outside it by two without det(J ∓ I) changing sign.
    out_prev = count(l -> abs(l) > 1.0, prev_eig)
    out_now = count(l -> abs(l) > 1.0, eig)
    complex_now = any(l -> abs(imag(l)) > 1.0e-12 * (1 + abs(real(l))), eig)
    if abs(out_now - out_prev) == 2 && complex_now && !(fold_turn || det_minus)
        push!(events, :neimark_sacker)
    end
    return events
end

"""
    trace_cycle_arclength(map, y0, params::NamedTuple, param_key::Symbol;
                          period = 1, direction = 1, bounds = nothing,
                          step = 0.01, step_min = 1e-6, step_max = 0.1, max_steps = 1000,
                          newton_tol = 1e-10, newton_max_iter = 10,
                          show_progress = false, callback = nothing) -> Vector{ContinuationStep}

Follow a branch of periodic points by pseudo-arclength continuation.

Unlike [`trace_cycle`](@ref), which prescribes the parameter values, this
continuation treats the periodic point and the parameter together as a point
``u = (y, λ)`` on the curve ``T^n(y, λ) - y = 0`` and advances along the
curve itself, so folds (saddle-node bifurcations), where the branch turns
back in ``λ``, are passed and the continuation carries on along the other
side. The tangent of the curve is the null vector of
``[DT^n - I \\mid ∂T^n/∂λ]``, available analytically from `MapResult.jac` and
`MapResult.jac_p`.

Bifurcations along the branch are detected from sign changes between
consecutive points and recorded in `info[:events]` of the later point:

* `:fold`: the tangent's parameter component changes sign;
* `:branch_point`: ``\\det(DT^n - I)`` changes sign without a fold
  (transcritical or pitchfork; the continuation follows one branch);
* `:period_doubling`: ``\\det(DT^n + I)`` changes sign;
* `:neimark_sacker`: a complex pair of multipliers crosses the unit circle.

`direction` (`+1`: ``λ`` increasing first) and `bounds` (`(lo, hi)` on ``λ``)
work as in [`arclength_continuation`](@ref), as do `step`, `step_min`,
`step_max`, `max_steps`, `newton_tol` and `newton_max_iter`. Each step's `info` holds `:eigvals`, `:eigvecs`,
`:tangent` (unit vector in the ``(y, λ)`` space), `:step`,
`:newton_iterations` and `:events`.

# Examples

The fixed point of `y ↦ y + p - y²` exists for `p ≥ 0` only; both fixed
points `±√p` lie on one curve that folds at `p = 0`:

```julia
f = Mapping((y, p) -> y[1] + p.p - y[1]^2, 1)
steps = trace_cycle_arclength(Diffeomorphism(f), [0.5], (p = 0.25,), :p;
    direction = -1, step = 0.05, step_max = 0.1, bounds = (-1.0, 0.25))
any(:fold in s.info[:events] for s in steps)         # true
steps[end].y[1] < 0 && steps[end].params.p > 0       # true: the lower branch
```
"""
function trace_cycle_arclength(
        map::AbstractMap, y0, params::NamedTuple, param_key::Symbol;
        period::Integer = 1, direction::Integer = 1, bounds = nothing,
        step::Real = 0.01, step_min::Real = 1.0e-6, step_max::Real = 0.1, max_steps::Integer = 1000,
        newton_tol::Real = 1.0e-10, newton_max_iter::Integer = 10,
        show_progress::Bool = false, callback = nothing,
    )
    haskey(params, param_key) || throw(ArgumentError("the map does not depend on parameter $(repr(param_key)): it is not among the parameters $(keys(params))"))
    col = findfirst(==(param_key), keys(params))
    y = as_state(y0)
    dim = length(y)
    last = Ref{Union{Nothing, MapResult}}(nothing)
    with_param(value) = merge(params, NamedTuple{(param_key,)}((Float64(value),)))

    system = function (u)
        res = image_detail(map, u[1:dim], with_param(u[dim + 1]); n = period, jacobian = true, parameter_jacobian = true)
        last[] = res
        return res.y - u[1:dim], hcat(res.jac - I, res.jac_p[:, col:col])
    end

    steps = ContinuationStep[]
    prev_tangent = Ref(Float64[])
    prev_jac = Ref(zeros(0, 0))
    prev_eig = Ref(ComplexF64[])
    on_point = function (point::ArclengthPoint)
        # `system` was last evaluated at the accepted point by the corrector.
        res = last[]
        eig = LinearAlgebra.eigvals(res)
        eigvec = LinearAlgebra.eigvecs(res)
        events = isempty(steps) ? Symbol[] : _classify_events(prev_tangent[], point.tangent, prev_jac[], res.jac, prev_eig[], eig)
        prev_tangent[] = point.tangent
        prev_jac[] = res.jac
        prev_eig[] = eig
        info = Dict{Symbol, Any}(
            :eigvals => eig, :eigvecs => eigvec, :tangent => point.tangent,
            :step => point.step, :newton_iterations => point.newton_iterations, :events => events,
        )
        cs = ContinuationStep(point.u[1:dim], with_param(point.u[dim + 1]), info)
        push!(steps, cs)
        callback === nothing || callback(cs)
        if show_progress
            tag = isempty(events) ? "" : " " * join(string.(events), ",")
            print(stderr, "ARCLENGTH ", lpad(length(steps), 5), " ", param_key, "=", Printf.@sprintf("%+.10g", cs.params[param_key]), " h=", Printf.@sprintf("%.3g", point.step), " y=", cs.y, tag, "\r")
            flush(stderr)
        end
        return nothing
    end

    try
        arclength_continuation(
            system, vcat(y, Float64(params[param_key]));
            step, step_min, step_max, max_steps, direction, bounds, newton_tol, newton_max_iter, callback = on_point,
        )
    finally
        show_progress && println(stderr)
    end
    return steps
end
