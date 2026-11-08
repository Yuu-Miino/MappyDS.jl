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
struct StepFunctionResult{T}
    success::Bool
    y::Union{Nothing, T}
    params::Union{Nothing, NamedTuple}
    info::Union{Nothing, Dict{Symbol, Any}}
end

function StepFunctionResult(success::Bool, y = nothing, params = nothing, info = nothing)
    return StepFunctionResult{typeof(y)}(success, y, params, info)
end

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
