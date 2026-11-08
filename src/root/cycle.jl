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
