# System functions: a user-written `f(y, p)` wrapped with its dimensions,
# evaluated on static vectors and differentiated with ForwardDiff.

"""
    SystemFunction

Abstract supertype of [`VectorField`](@ref), [`Mapping`](@ref) and
[`Border`](@ref): a user function `f(y, p)` of a state vector `y` and
parameters `p` together with the dimensions of its domain and codomain.

The function is evaluated on `SVector`s (StaticArrays) and differentiated
with ForwardDiff, so it must be written generically: build the result with
`SVector(...)`, `[...]` or a tuple, do not fix element types to `Float64`,
and read parameters as `p.a` (a `NamedTuple`) or any other container you
pass. Branches on the values of `y` are allowed (unlike the SymPy tracing of
the Python package), but the function must be differentiable where it is
evaluated.

See also [`jacobian`](@ref), [`hessian`](@ref), [`dom_dim`](@ref),
[`cod_dim`](@ref).
"""
abstract type SystemFunction end

_default_name(f) = f isa Function ? nameof(f) : nameof(typeof(f))

"""
    VectorField(f, dim; name = nameof(f))

Declare the right-hand side of an autonomous ODE ``\\dot y = f(y, p)`` on
``\\mathbb{R}^{dim}`` (Python: `@vector_field(dim)`).

# Examples

```julia
izhikevich = VectorField(2) do y, p
    v, u = y
    return SVector(0.04 * v^2 + 5 * v + 140 - u + p.I, p.a * (p.b * v - u))
end
jacobian(izhikevich, [0.0, 0.0], (a = 0.2, b = 0.2, I = 10.0))
```
"""
struct VectorField{N, F} <: SystemFunction
    f::F
    name::Symbol
end

function VectorField(f, dim::Integer; name = _default_name(f))
    dim >= 1 || throw(ArgumentError("dim must be a positive integer, got $dim"))
    return VectorField{Int(dim), typeof(f)}(f, Symbol(name))
end

"""
    Mapping(f, dom_dim, cod_dim = dom_dim; name = nameof(f))

Declare a discrete map ``y \\mapsto f(y, p)`` from ``\\mathbb{R}^{dom\\_dim}``
to ``\\mathbb{R}^{cod\\_dim}`` (Python: `@mapping(dom_dim, cod_dim)`).

# Examples

```julia
henon = Mapping(2) do y, p
    x, u = y
    return SVector(1 - p.a * x^2 + u, p.b * x)
end
henon([0.0, 0.0], (a = 1.4, b = 0.3))  # [1.0, 0.0]
```
"""
struct Mapping{N, M, F} <: SystemFunction
    f::F
    name::Symbol
end

function Mapping(f, dom_dim::Integer, cod_dim::Integer = dom_dim; name = _default_name(f))
    (dom_dim >= 1 && cod_dim >= 1) || throw(ArgumentError("dom_dim and cod_dim must be positive integers"))
    return Mapping{Int(dom_dim), Int(cod_dim), typeof(f)}(f, Symbol(name))
end

"""
    Border(q, dim; direction = 1, name = nameof(q))

Declare a border (section) ``q(y, p) = 0`` of a continuous mode on
``\\mathbb{R}^{dim}`` (Python: `@border(dim, direction, name)`). `direction`
is the zero crossing that terminates the mode: `1` for negative to positive,
`-1` for positive to negative and `0` for both. `name` identifies the border
in transition rules written as `Dict(:mode => Dict(:border => :next))`.
"""
struct Border{N, F} <: SystemFunction
    f::F
    name::Symbol
    direction::Int
end

function Border(f, dim::Integer; direction::Integer = 1, name = _default_name(f))
    dim >= 1 || throw(ArgumentError("dim must be a positive integer, got $dim"))
    direction in (-1, 0, 1) || throw(ArgumentError("direction must be 1, -1 or 0, got $direction"))
    return Border{Int(dim), typeof(f)}(f, Symbol(name), Int(direction))
end

"""
    dom_dim(f)

Dimension of the domain of a system function, mode or map.
"""
dom_dim(::VectorField{N}) where {N} = N
dom_dim(::Mapping{N}) where {N} = N
dom_dim(::Border{N}) where {N} = N

"""
    cod_dim(f)

Dimension of the codomain of a system function or mode.
"""
cod_dim(::VectorField{N}) where {N} = N
cod_dim(::Mapping{N, M}) where {N, M} = M
cod_dim(::Border) = 1

"""
    direction(b::Border)

Direction of the zero crossing that triggers the border (`1`, `-1` or `0`).
"""
direction(b::Border) = b.direction

Base.nameof(f::SystemFunction) = f.name

Base.show(io::IO, f::VectorField) = print(io, "VectorField(", f.name, ", dim = ", dom_dim(f), ")")
Base.show(io::IO, f::Mapping) = print(io, "Mapping(", f.name, ", dom_dim = ", dom_dim(f), ", cod_dim = ", cod_dim(f), ")")
Base.show(io::IO, f::Border) = print(io, "Border(", f.name, ", dim = ", dom_dim(f), ", direction = ", f.direction, ")")

# -- state conversion ---------------------------------------------------------

"""
    as_state(y) -> Vector{Float64}

Convert `y` (a number, tuple or vector) to a one-dimensional `Float64`
vector; array inputs are copied.
"""
as_state(y::Real) = Float64[y]
as_state(y::Tuple) = collect(Float64, y)
as_state(y::AbstractVector) = Vector{Float64}(y)
as_state(y::AbstractArray) = throw(ArgumentError("state must be one-dimensional, got size $(size(y))"))
as_state(y) = throw(ArgumentError("cannot convert $(typeof(y)) to a state vector"))

@inline function _head(::Val{N}, z::AbstractVector) where {N}
    return SVector{N, Float64}(ntuple(i -> @inbounds(Float64(z[i])), Val(N)))
end

function _svec(f::SystemFunction, y)
    n = dom_dim(f)
    v = as_state(y)
    length(v) == n || throw(ArgumentError("$(repr(f.name)) expects a state of length $n, got $(length(v))"))
    return _head(Val(n), v)
end

@inline _to_svector(::Val{M}, out::Number) where {M} = M == 1 ? SVector{1}(out) : throw(DimensionMismatch("expected $M components, got a scalar"))
@inline function _to_svector(::Val{M}, out) where {M}
    length(out) == M || throw(DimensionMismatch("expected $M components, got $(length(out))"))
    return SVector{M}(out)
end

# -- evaluation (hot path, static arrays) ----------------------------------------

@inline _eval(f::SystemFunction, y::SVector, p) = _to_svector(Val(cod_dim(f)), f.f(y, p))

@inline function _jac(f::SystemFunction, y::SVector{N}, p) where {N}
    return ForwardDiff.jacobian(yy -> _eval(f, yy, p), y)
end

@inline function _hes(f::SystemFunction, y::SVector{N}, p) where {N}
    M = cod_dim(f)
    h = ForwardDiff.jacobian(yy -> vec(_jac(f, yy, p)), y)
    return SArray{Tuple{M, N, N}, eltype(h)}(Tuple(h))
end

# -- parameter derivatives (hot path) -------------------------------------------

_nparams(::Nothing) = 0
_nparams(::NamedTuple{K}) where {K} = length(K)
_nparams(p) = throw(ArgumentError("parameter derivatives require the parameters as a NamedTuple of real numbers, got $(typeof(p))"))

# `∂f/∂p` as a `cod_dim × k` static matrix, columns in the order of `keys(p)`.
@inline function _jac_p(f::SystemFunction, y::SVector, p::NamedTuple{K}) where {K}
    M = cod_dim(f)
    length(K) == 0 && return zeros(SMatrix{M, 0, Float64})
    pv = SVector{length(K), Float64}(values(p))
    return ForwardDiff.jacobian(pp -> _eval(f, y, NamedTuple{K}(Tuple(pp))), pv)
end
_jac_p(f::SystemFunction, ::SVector, ::Nothing) = zeros(SMatrix{cod_dim(f), 0, Float64})
_jac_p(::SystemFunction, ::SVector, p) = _nparams(p)

# `∂²f/∂y∂p` as a `cod_dim × dom_dim × k` static array (nested ForwardDiff).
@inline function _hes_p(f::SystemFunction, y::SVector{N}, p::NamedTuple{K}) where {N, K}
    M = cod_dim(f)
    length(K) == 0 && return zeros(SArray{Tuple{M, N, 0}, Float64})
    pv = SVector{length(K), Float64}(values(p))
    h = ForwardDiff.jacobian(pp -> vec(_jac(f, y, NamedTuple{K}(Tuple(pp)))), pv)
    return SArray{Tuple{M, N, length(K)}, eltype(h)}(Tuple(h))
end
_hes_p(f::SystemFunction, ::SVector{N}, ::Nothing) where {N} = zeros(SArray{Tuple{cod_dim(f), N, 0}, Float64})
_hes_p(::SystemFunction, ::SVector, p) = _nparams(p)

# -- public evaluation --------------------------------------------------------

"""
    (f::SystemFunction)(y, p = nothing) -> Vector{Float64}

Evaluate the function numerically.
"""
(f::SystemFunction)(y, p = nothing) = Vector(_eval(f, _svec(f, y), p))

"""
    jacobian(f::SystemFunction, y, p = nothing) -> Matrix{Float64}

Jacobian matrix `jac[i, j] = ∂f_i/∂y_j` at `y`, computed with ForwardDiff.
"""
jacobian(f::SystemFunction, y, p = nothing) = Matrix(_jac(f, _svec(f, y), p))

"""
    hessian(f::SystemFunction, y, p = nothing) -> Array{Float64, 3}

Hessian tensor `hes[i, j, k] = ∂²f_i/(∂y_j ∂y_k)` at `y`, computed with
nested ForwardDiff Jacobians.
"""
hessian(f::SystemFunction, y, p = nothing) = Array(_hes(f, _svec(f, y), p))

"""
    parameter_jacobian(f::SystemFunction, y, p) -> Matrix{Float64}

Derivative `jac_p[i, m] = ∂f_i/∂p_m` at `y` with respect to the parameters,
one column per key of the `NamedTuple` `p` in its order (a `cod_dim × 0`
matrix when `p` is `nothing` or empty). Parameters the function does not
read give zero columns.
"""
parameter_jacobian(f::SystemFunction, y, p = nothing) = Matrix(_jac_p(f, _svec(f, y), p))

"""
    mixed_hessian(f::SystemFunction, y, p) -> Array{Float64, 3}

Mixed second derivative `hes_p[i, j, m] = ∂²f_i/(∂y_j ∂p_m)` at `y`, the
last axis in the order of the keys of the `NamedTuple` `p`.
"""
mixed_hessian(f::SystemFunction, y, p = nothing) = Array(_hes_p(f, _svec(f, y), p))
