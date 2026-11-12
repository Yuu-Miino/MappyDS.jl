# Per-operation timings of MappyDS.jl on the benchmark systems.
#
# Mirrors bench.py: the same systems, tolerances and operation counts, so that
# the per-call times are comparable. Writes a JSON file (ARGS[1]) with
# {task: {"per_call_ms": float, "n": int}}; the first-call (compile) times are
# written to {task}_first_call_ms as well.
#
# Run: julia --project=bench bench/bench.jl out.json [scale]

using MappyDS
using StaticArrays: SVector
using Statistics: median
using JSON3

const SCALE = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
const REPEATS = 3
n_of(n) = max(1, round(Int, n * SCALE))

# -- systems (as in the test suites of both packages) --------------------------

izhikevich = VectorField(2; name = :izhikevich) do y, p
    v, u = y
    return SVector(0.04 * v^2 + 5 * v + 140 - u + p.I, p.a * (p.b * v - u))
end
fire = Border((y, _) -> y[1] - 30, 2; name = :fire)
firing = Mapping((y, p) -> y + SVector(p.c - 30, p.d), 2; name = :firing)
reduce_u = Mapping((y, _) -> y[2], 2, 1; name = :reduce_u)
revert = Mapping((u, _) -> SVector(30.0, u[1]), 1, 2; name = :revert)

make_izhikevich_map() = PoincareMap(
    [
        ContinuousMode(:A, izhikevich; borders = (fire,), reltol = 1.0e-6, abstol = 1.0e-9),
        DiscreteMode(:B, firing),
        DiscreteMode(:P, reduce_u),
        DiscreteMode(:Pinv, revert),
    ],
    Dict(:A => [:P], :B => :A, :P => :Pinv, :Pinv => :B),
    :Pinv,
)

henon = Mapping(2; name = :henon) do y, p
    x, u = y
    return SVector(1 - p.a * x^2 + u, p.b * x)
end

duffing = VectorField(3; name = :duffing) do y, p
    x, v, t = y
    return SVector(v, -p.k * v - x^3 + p.B0 + p.B * cos(t), one(t))
end
duffing_section = Border((y, _) -> y[3] - 2pi, 3; name = :duffing_section)
duffing_t_reset = Mapping((y, _) -> y - SVector(0, 0, 2pi), 3; name = :duffing_t_reset)
append_time = Mapping((y, _) -> SVector(y[1], y[2], zero(y[1])), 2, 3; name = :append_time)
drop_time = Mapping((y, _) -> SVector(y[1], y[2]), 3, 2; name = :drop_time)

make_duffing_map() = PoincareMap(
    [
        ContinuousMode(:duffing, duffing; borders = (duffing_section,), reltol = 1.0e-6, abstol = 1.0e-9),
        DiscreteMode(:t_reset, duffing_t_reset),
        DiscreteMode(:p, append_time),
        DiscreteMode(:pinv, drop_time),
    ],
    Dict(:p => :duffing, :duffing => [:t_reset], :t_reset => :pinv, :pinv => :p),
    :p,
)

const IZH = (a = 0.2, b = 0.2, c = -50.0, d = 2.0, I = 10.0)
const IZH8 = merge(IZH, (I = 8.0,))
const HEN = (a = 1.4, b = 0.3)
const DUF = (k = 0.2, B0 = 0.1, B = 0.1)
const DUF_Y0 = [0.232147751858, 0.075591687407]

const results = Dict{String, Any}()

"""Time `fn()` called n times; report the median per-call time of REPEATS batches."""
function bench(name, n, fn)
    t0 = time_ns()
    fn()  # first call: compilation
    first_ms = (time_ns() - t0) / 1.0e6
    per_call = Float64[]
    for _ in 1:REPEATS
        t0 = time_ns()
        for _ in 1:n
            fn()
        end
        push!(per_call, (time_ns() - t0) / 1.0e6 / n)
    end
    results[name] = Dict("per_call_ms" => median(per_call), "n" => n, "first_call_ms" => first_ms)
    println(rpad(name, 28), lpad(round(median(per_call); digits = 4), 12), " ms  (n=$n, first call $(round(first_ms; digits = 1)) ms)")
    return nothing
end

hmap = Diffeomorphism(henon)
izh = make_izhikevich_map()
duf = make_duffing_map()
y_h = [0.1, 0.1]

# Discrete map: one evaluation
bench("henon_image", n_of(20000), () -> image_detail(hmap, y_h, HEN; jacobian = false))
bench("henon_jac_hes", n_of(5000), () -> image_detail(hmap, y_h, HEN; hessian = true))
bench("henon_full", n_of(2000), () -> image_detail(hmap, y_h, HEN; hessian = true, parameter_jacobian = true, parameter_hessian = true))

# Poincaré map of the Izhikevich neuron (1-D section, one firing per iteration)
bench("izh_image", n_of(300), () -> image_detail(izh, -3.0, IZH; jacobian = false))
bench("izh_jac", n_of(300), () -> image_detail(izh, -3.0, IZH))
bench("izh_jac_hes", n_of(200), () -> image_detail(izh, -3.0, IZH; hessian = true))
bench("izh_full", n_of(100), () -> image_detail(izh, -3.0, IZH; hessian = true, parameter_jacobian = true, parameter_hessian = true))

# Stroboscopic map of the forced Duffing oscillator (one period 2π)
bench("duffing_image", n_of(100), () -> image_detail(duf, DUF_Y0, DUF; jacobian = false))
bench("duffing_jac_hes", n_of(60), () -> image_detail(duf, DUF_Y0, DUF; hessian = true))
bench("duffing_full", n_of(30), () -> image_detail(duf, DUF_Y0, DUF; hessian = true, parameter_jacobian = true, parameter_hessian = true))

# Periodic points, continuation, local bifurcations
bench("izh_find_cycle", n_of(20), () -> find_cycle(izh, -4.6, IZH8))
fp = find_cycle(izh, -4.6, IZH8).y
n_trace = 20  # fixed: fewer points would make the continuation steps too large
bench("izh_trace_cycle_per_point", 1, () -> trace_cycle(izh, fp, IZH8, :I, 4.5; resolution = n_trace))
results["izh_trace_cycle_per_point"]["per_call_ms"] /= n_trace
results["izh_trace_cycle_per_point"]["n"] = n_trace

duf_steps = trace_cycle(duf, find_cycle(duf, DUF_Y0, DUF).y, DUF, :B, 0.17; resolution = 10)
last_step = duf_steps[end]
bench("duffing_find_local_bf", n_of(5), () -> find_local_bf(duf, last_step.y, last_step.params, :B, pi))

hp = (a = 0.35, b = 0.3)
hfp = find_cycle(hmap, [0.9, 0.3], hp).y
hbf = find_local_bf(hmap, hfp, hp, :a, pi)
n_bf = 100
bench("henon_trace_local_bf_per_point", 1, () -> trace_local_bf(hmap, hbf.y, hbf.params, :a, hbf.theta, :b, 0.2; resolution = n_bf))
results["henon_trace_local_bf_per_point"]["per_call_ms"] /= n_bf
results["henon_trace_local_bf_per_point"]["n"] = n_bf

open(ARGS[1], "w") do io
    JSON3.pretty(io, results)
end
println("wrote ", ARGS[1])
