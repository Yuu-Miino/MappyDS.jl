# Natural-parameter continuation driven by a user-supplied step function.

"""
    ContinuationStep(y, params, info)

One successful step of a parameter continuation: the continued variable
`y`, the parameters `params` (a `NamedTuple`) at this step and extra data
`info` (a `Dict{Symbol, Any}`, empty if none) returned by the step function.
"""
struct ContinuationStep{T}
    y::T
    params::NamedTuple
    info::Dict{Symbol, Any}
end

"""
    StepFunctionResult(success, y = nothing, params = nothing, info = nothing)

Return value of a step function passed to [`continuation`](@ref). The
continuation stops at the first step with `success == false` or
`y === nothing`. `params` replaces the parameters for the following steps
(e.g. with a solved parameter updated); `info` is stored in
[`ContinuationStep`](@ref).
"""
struct StepFunctionResult
    success::Bool
    y::Any
    params::Union{Nothing, NamedTuple}
    info::Union{Nothing, Dict{Symbol, Any}}
end

StepFunctionResult(success::Bool) = StepFunctionResult(success, nothing, nothing, nothing)
StepFunctionResult(success::Bool, y) = StepFunctionResult(success, y, nothing, nothing)
StepFunctionResult(success::Bool, y, params) = StepFunctionResult(success, y, params, nothing)

"""
    continuation(fun, y0, params::NamedTuple, param_key::Symbol, end_value;
                 resolution = 100, show_progress = false, callback = nothing)

Continue the solution of a step function along a parameter.

`params[param_key]` is swept linearly from its current value to `end_value`
in `resolution` points, both ends included. At each point `fun(y, p)` is
called, where `y` and `p` are those of the previous successful step (`y0`
and `params` at the first point) with `p[param_key]` set to the current
value; it must return a [`StepFunctionResult`](@ref). The sweep stops at the
first unsuccessful step. `callback` is called with each successful
[`ContinuationStep`](@ref); `show_progress` prints one status line per step
to `stderr`.

# Examples

```julia
step(y, p) = StepFunctionResult(true, sqrt(p.p))
steps = continuation(step, 1.0, (p = 1.0,), :p, 4.0; resolution = 4)
[(s.params.p, s.y) for s in steps]  # (1.0, 1.0), (2.0, 1.414…), (3.0, 1.732…), (4.0, 2.0)
```
"""
function continuation(fun, y0, params::NamedTuple, param_key::Symbol, end_value::Real; resolution::Integer = 100, show_progress::Bool = false, callback = nothing)
    resolution >= 2 || throw(ArgumentError("resolution must be at least 2, got $resolution"))
    haskey(params, param_key) || throw(ArgumentError("parameter $(repr(param_key)) is not in params"))
    values = range(Float64(params[param_key]), Float64(end_value); length = Int(resolution))
    width = length(string(resolution))
    y = y0
    p = params
    found = ContinuationStep[]
    try
        for (i, value) in enumerate(values)
            p = merge(p, NamedTuple{(param_key,)}((Float64(value),)))
            ret = fun(y, p)
            (ret.success && ret.y !== nothing) || break
            y = ret.y
            if ret.params !== nothing
                p = ret.params
            end
            step = ContinuationStep(y, p, ret.info === nothing ? Dict{Symbol, Any}() : copy(ret.info))
            push!(found, step)
            callback === nothing || callback(step)
            if show_progress
                print(stderr, "SUCCESS ", lpad(i, width), "/", resolution, " ", param_key, "=", Printf.@sprintf("%+.10g", p[param_key]), " y=", y, "\r")
                flush(stderr)
            end
        end
    finally
        show_progress && println(stderr)
    end
    return found
end

# -- pseudo-arclength continuation -------------------------------------------------

"""
    ArclengthPoint

One accepted point of [`arclength_continuation`](@ref): the point `u` on
the solution curve (length `m + 1`), the unit `tangent` of the curve at `u`
in the direction of travel, the arclength `step` that led to this point
(`0` for the first point) and the `newton_iterations` the corrector needed.
"""
struct ArclengthPoint
    u::Vector{Float64}
    tangent::Vector{Float64}
    step::Float64
    newton_iterations::Int
end

const _RETRYABLE = Union{MappyError, LinearAlgebra.SingularException, LinearAlgebra.LAPACKException, DomainError}

# Unit null vector of `jac` oriented like `previous`: with a previous tangent
# the bordered system `[J; previousᵀ] t = e_last` is solved, which keeps the
# orientation; without one the null vector comes from the SVD and is oriented
# so that its last component has the sign of `direction`.
function _tangent(jac::AbstractMatrix, previous, direction::Int)
    m = size(jac, 1)
    if previous !== nothing
        bordered = vcat(jac, permutedims(previous))
        rhs = zeros(m + 1)
        rhs[end] = 1.0
        t = bordered \ rhs
    else
        t = svd(jac; full = true).Vt[end, :]
        if t[end] * direction < 0
            t = -t
        end
    end
    return t / norm(t)
end

# Newton on the bordered system; `nothing` if it does not converge. The last
# evaluation of `fun` is at the returned point.
function _arclength_corrector(fun, u_pred::Vector{Float64}, base::Vector{Float64}, t::Vector{Float64}, h::Float64, m::Int, newton_tol::Float64, newton_max_iter::Int)
    x = copy(u_pred)
    for k in 1:newton_max_iter
        res, jac = fun(x)
        res = Vector{Float64}(vec(res))
        jac = Matrix{Float64}(reshape(jac, m, m + 1))
        g = vcat(res, dot(t, x - base) - h)
        bordered = vcat(jac, permutedims(t))
        dx = bordered \ (-g)
        x = x + dx
        if norm(dx) <= newton_tol * (1.0 + norm(x))
            res, jac = fun(x)
            res = Vector{Float64}(vec(res))
            jac = Matrix{Float64}(reshape(jac, m, m + 1))
            if norm(res) <= 100 * newton_tol * (1.0 + norm(x))
                return x, jac, k
            end
            return nothing
        end
    end
    return nothing
end

function _try_corrector(args...)
    return try
        _arclength_corrector(args...)
    catch err
        err isa _RETRYABLE || rethrow()
        nothing
    end
end

"""
    arclength_continuation(fun, u0; step = 0.01, step_min = 1e-6, step_max = 0.1,
                           max_steps = 1000, direction = 1, bounds = nothing,
                           newton_tol = 1e-10, newton_max_iter = 10, callback = nothing)

Follow a solution curve of ``F(u) = 0`` by pseudo-arclength continuation.

`fun(u)` returns the residual ``F(u)`` (length `m`) and its Jacobian
(`m × (m + 1)`); ``F : \\mathbb{R}^{m+1} \\to \\mathbb{R}^m`` has a
one-dimensional zero set. Starting from `u0` (first corrected onto the
curve with its last component held fixed), each step predicts
``u = u_k + h t_k`` along the unit tangent ``t_k`` and corrects it with
Newton's method on the bordered system ``F(u) = 0``, ``t_k \\cdot (u - u_k) = h``,
whose Jacobian ``[J; t_k^T]`` is regular even where ``J`` loses rank in the
parameter direction, so folds are passed without special treatment
(Keller, 1977). The step grows by 1.5 when Newton converges in at most three
iterations, is halved when six or more are needed or when it fails, and the
orientation is kept by requiring ``t_k \\cdot t_{k+1} > 0``.

The last component of `u` is the parameter-like coordinate used by
`direction` (`+1` to start with it increasing, `-1` decreasing) and `bounds`
(`(lo, hi)`: stop when it would leave the interval; the last step is
shortened so that the final point lies close to the boundary). `fun` may
throw a [`MappyError`](@ref), a `SingularException` or a `DomainError` to
signal that `u` is not evaluable; the step is then retried with a smaller
`h`. The continuation stops when `bounds` are left, `max_steps` is reached,
the step cannot be reduced below `step_min`, or the curve closes on itself;
it throws an `ErrorException` if `u0` cannot be corrected onto the curve.

The arclength is measured in the Euclidean metric of `u`; if the state and
the parameter have very different scales, rescale the parameter in `fun`.

# Examples

```julia
circle(u) = ([dot(u, u) - 1.0], permutedims(2u))
pts = arclength_continuation(circle, [1.0, 0.0]; step = 0.3, step_max = 0.3)
all(abs(dot(p.u, p.u) - 1) < 1e-9 for p in pts)  # true; stops after one turn
```
"""
function arclength_continuation(
        fun, u0;
        step::Real = 0.01, step_min::Real = 1.0e-6, step_max::Real = 0.1, max_steps::Integer = 1000,
        direction::Integer = 1, bounds = nothing, newton_tol::Real = 1.0e-10, newton_max_iter::Integer = 10,
        callback = nothing,
    )
    u = Vector{Float64}(vec(collect(Float64, u0)))
    length(u) >= 2 || throw(ArgumentError("u0 must be a vector of length m + 1 >= 2"))
    direction in (1, -1) || throw(ArgumentError("direction must be +1 or -1"))
    (0 < step_min <= step <= step_max) || throw(ArgumentError("need 0 < step_min <= step <= step_max"))
    bounds === nothing || length(bounds) == 2 || throw(ArgumentError("bounds must be (lo, hi)"))
    m = length(u) - 1
    tol = Float64(newton_tol)
    iters_max = Int(newton_max_iter)
    dir = Int(direction)

    # Correct the starting point with its parameter held fixed.
    e_last = zeros(m + 1)
    e_last[end] = 1.0
    start = _try_corrector(fun, u, u, e_last, 0.0, m, tol, iters_max)
    start === nothing && error("the starting point could not be corrected onto the curve")
    u, jac, iters = start
    t = _tangent(jac, nothing, dir)
    points = ArclengthPoint[ArclengthPoint(u, t, 0.0, iters)]
    callback === nothing || callback(points[1])

    h = Float64(step)
    hmin = Float64(step_min)
    hmax = Float64(step_max)
    u_start = copy(u)
    for _ in 1:max_steps
        accepted = nothing
        while accepted === nothing
            h < hmin && return points
            accepted = _try_corrector(fun, u + h * t, u, t, h, m, tol, iters_max)
            accepted === nothing && (h *= 0.5)
        end
        u_new, jac_new, iters = accepted
        bounds_hit = false
        if bounds !== nothing && !(bounds[1] <= u_new[end] <= bounds[2])
            # Shrink the step so that the next point lands close to the
            # boundary; give up after a few attempts (e.g. strong curvature).
            landed = false
            for _ in 1:4
                target = u_new[end] < bounds[1] ? Float64(bounds[1]) : Float64(bounds[2])
                frac = (target - u[end]) / (u_new[end] - u[end])
                h = max(0.999 * frac * h, hmin)
                h <= hmin && break
                accepted = _try_corrector(fun, u + h * t, u, t, h, m, tol, iters_max)
                accepted === nothing && break
                u_new, jac_new, iters = accepted
                if bounds[1] <= u_new[end] <= bounds[2]
                    landed = true
                    break
                end
            end
            landed || return points
            bounds_hit = true
        end
        t_new = try
            _tangent(jac_new, t, dir)
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            _tangent(jac_new, nothing, t[end] >= 0 ? 1 : -1)
        end
        point = ArclengthPoint(u_new, t_new, h, iters)
        push!(points, point)
        callback === nothing || callback(point)
        bounds_hit && return points
        # Closed curve: back at the start after having moved away from it.
        length(points) > 3 && norm(u_new - u_start) < 0.5 * h && return points
        u, t = u_new, t_new
        if iters <= 3
            h = min(1.5 * h, hmax)
        elseif iters >= 6
            h = max(0.5 * h, hmin)
        end
    end
    return points
end
