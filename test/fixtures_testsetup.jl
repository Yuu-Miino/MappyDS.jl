# Shared fixtures: the systems used throughout the test suite (as in the
# Python package's tests/conftest.py) and the reference values exported from
# the Python implementation (test/fixtures/reference.json).
@testsetup module Fixtures

using MappyDS
using StaticArrays: SVector
using OrdinaryDiffEqVerner: Vern9
using JSON3

export fd_jacobian, fd_hessian, reference, refvec, refmat, reften, refparams, refcomplex
export izhikevich, fire, firing, reduce_u, revert, make_izhikevich_map, IZH_PARAMS, IZH_Y0_MODE
export henon, HENON_PARAMS, henon_map
export duffing, make_duffing_map, DUFFING_PARAMS, DUFFING_Y0
export make_relay_map, RELAY_PARAMS
export TIGHT

# -- finite-difference helpers ------------------------------------------------

function fd_jacobian(f, y0::AbstractVector, h = 1.0e-4)
    y0 = Vector{Float64}(y0)
    cols = map(1:length(y0)) do j
        e = zeros(length(y0))
        e[j] = h
        return (f(y0 + e) - f(y0 - e)) / (2h)
    end
    return reduce(hcat, cols)
end

function fd_hessian(f, y0::AbstractVector, h = 1.0e-4)
    y0 = Vector{Float64}(y0)
    n = length(y0)
    m = length(f(y0))
    hes = zeros(m, n, n)
    for j in 1:n, k in 1:n
        ej = zeros(n)
        ej[j] = h
        ek = zeros(n)
        ek[k] = h
        hes[:, j, k] = (f(y0 + ej + ek) - f(y0 + ej - ek) - f(y0 - ej + ek) + f(y0 - ej - ek)) / (4h * h)
    end
    return hes
end

# -- reference values ---------------------------------------------------------

_plain(x::JSON3.Object) = Dict{String, Any}(String(k) => _plain(v) for (k, v) in x)
_plain(x::JSON3.Array) = Any[_plain(v) for v in x]
_plain(x) = x
const reference = _plain(JSON3.read(read(joinpath(@__DIR__, "fixtures", "reference.json"), String)))

refvec(x) = Float64[v for v in x]
refmat(x) = Float64[x[i][j] for i in eachindex(x), j in eachindex(x[1])]
reften(x) = Float64[x[i][j][k] for i in eachindex(x), j in eachindex(x[1]), k in eachindex(x[1][1])]
refparams(d) = (; (Symbol(k) => Float64(v) for (k, v) in d)...)
refcomplex(x) = ComplexF64[complex(Float64(z[1]), Float64(z[2])) for z in x]

# High-accuracy solver settings used where the Python tests use rtol=1.0e-10, atol=1.0e-12.
const TIGHT = (alg = Vern9(), reltol = 1.0e-10, abstol = 1.0e-12)

# -- Izhikevich neuron ------------------------------------------------------------

const izhikevich = VectorField(2; name = :izhikevich) do y, p
    v, u = y
    return SVector(0.04 * v^2 + 5 * v + 140 - u + p.I, p.a * (p.b * v - u))
end
const fire = Border((y, _) -> y[1] - 30, 2; name = :fire)
const firing = Mapping((y, p) -> y + SVector(p.c - 30, p.d), 2; name = :firing)
const reduce_u = Mapping((y, _) -> y[2], 2, 1; name = :reduce)
const revert = Mapping((u, _) -> SVector(30.0, u[1]), 1, 2; name = :revert)

const IZH_PARAMS = (a = 0.2, b = 0.2, c = -50.0, d = 2.0, I = 10.0)
const IZH_Y0_MODE = [-50.0, -1.7]

function make_izhikevich_map(; kwargs...)
    return PoincareMap(
        [
            ContinuousMode(:A, izhikevich; borders = (fire,), kwargs...),
            DiscreteMode(:B, firing),
            DiscreteMode(:P, reduce_u),
            DiscreteMode(:Pinv, revert),
        ],
        Dict(:A => [:P], :B => :A, :P => :Pinv, :Pinv => :B),
        :Pinv,
    )
end

# -- Hénon map ----------------------------------------------------------------------

const henon = Mapping(2; name = :henon) do y, p
    x, u = y
    return SVector(1 - p.a * x^2 + u, p.b * x)
end
const HENON_PARAMS = (a = 1.4, b = 0.3)
henon_map() = Diffeomorphism(henon)

# -- Forced Duffing oscillator ------------------------------------------------------

const duffing = VectorField(3; name = :duffing) do y, p
    x, v, t = y
    return SVector(v, -p.k * v - x^3 + p.B0 + p.B * cos(t), one(t))
end
const duffing_section = Border((y, _) -> y[3] - 2pi, 3; name = :duffing_section)
const duffing_t_reset = Mapping((y, _) -> y - SVector(0, 0, 2pi), 3; name = :duffing_t_reset)
const append_time = Mapping((y, _) -> SVector(y[1], y[2], zero(y[1])), 2, 3; name = :append_time)
const drop_time = Mapping((y, _) -> SVector(y[1], y[2]), 3, 2; name = :drop_time)

const DUFFING_PARAMS = (k = 0.2, B0 = 0.1, B = 0.1)
const DUFFING_Y0 = [0.232147751858, 0.075591687407]

function make_duffing_map(; kwargs...)
    return PoincareMap(
        [
            ContinuousMode(:duffing, duffing; borders = (duffing_section,), kwargs...),
            DiscreteMode(:t_reset, duffing_t_reset),
            DiscreteMode(:p, append_time),
            DiscreteMode(:pinv, drop_time),
        ],
        Dict(:p => :duffing, :duffing => [:t_reset], :t_reset => :pinv, :pinv => :p),
        :p,
    )
end

# -- Relay oscillator sampled with period T: two continuous modes, two end modes -----
#
# State (x, t). Mode :up: x' = 1 until x = 1 (-> :down); mode :down: x' = -1
# until x = -1 (-> :up); both stop at t = T (-> sampling section). With
# T = 2.2, starting in :up at x0 ∈ (-0.2, 1) the image is -0.2 - x0 in mode
# :down; starting in :down at x0 ∈ (-1, 0.2) it is 0.2 - x0 in mode :up.

const relay_up = VectorField((y, _) -> SVector(one(y[1]), one(y[1])), 2; name = :relay_up)
const relay_down = VectorField((y, _) -> SVector(-one(y[1]), one(y[1])), 2; name = :relay_down)
const upper = Border((y, _) -> y[1] - 1, 2; direction = 1, name = :upper)
const lower = Border((y, _) -> y[1] + 1, 2; direction = -1, name = :lower)
const sample = Border((y, p) -> y[2] - p.T, 2; name = :sample)
const time_reset = Mapping((y, _) -> SVector(y[1], zero(y[1])), 2; name = :time_reset)
const add_time = Mapping((y, _) -> SVector(y[1], zero(y[1])), 1, 2; name = :add_time)
const remove_time = Mapping((y, _) -> y[1], 2, 1; name = :remove_time)

const RELAY_PARAMS = (T = 2.2,)

function make_relay_map(; kwargs...)
    modes = [
        ContinuousMode(:up, relay_up; borders = (upper, sample), kwargs...),
        ContinuousMode(:down, relay_down; borders = (lower, sample), kwargs...),
        DiscreteMode(:r_up, time_reset),
        DiscreteMode(:r_down, time_reset),
        DiscreteMode(:p_up, add_time),
        DiscreteMode(:p_down, add_time),
        DiscreteMode(:pinv_up, remove_time),
        DiscreteMode(:pinv_down, remove_time),
    ]
    transitions = Dict(
        :p_up => :up,
        :p_down => :down,
        :up => [:down, :r_up],
        :down => [:up, :r_down],
        :r_up => :pinv_up,
        :r_down => :pinv_down,
        :pinv_up => :p_up,
        :pinv_down => :p_down,
    )
    return PoincareMap(modes, transitions, :p_up, [:p_up, :p_down])
end

end
