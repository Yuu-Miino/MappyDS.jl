@testitem "discrete mode evolve" setup = [Fixtures] begin
    mode = DiscreteMode(:H, henon)
    @test (dom_dim(mode), cod_dim(mode)) == (2, 2)
    @test occursin("H", repr(mode))
    res = evolve(mode, [0.5, 0.1], HENON_PARAMS)
    @test res isa StepResult
    @test res.y ≈ [1 - 1.4 * 0.25 + 0.1, 0.15]
    @test res.jac === nothing && res.hes === nothing
    @test res.border === nothing && res.t === nothing && res.segment === nothing
    res = evolve(mode, [0.5, 0.1], HENON_PARAMS; jacobian = true)
    @test res.jac ≈ [-1.4 1.0; 0.3 0.0]
    @test res.hes === nothing
    res = evolve(mode, [0.5, 0.1], HENON_PARAMS; hessian = true)
    @test res.jac !== nothing
    @test size(res.hes) == (2, 2, 2)
    @test res.hes[1, 1, 1] ≈ -2.8
end

@testitem "discrete segments need in_trajectory and trajectory" setup = [Fixtures] begin
    y0 = [0.5, 0.1]
    off = DiscreteMode(:H, henon)
    on = DiscreteMode(:H, henon; in_trajectory = true)
    @test evolve(off, y0, HENON_PARAMS; trajectory = true).segment === nothing
    @test evolve(on, y0, HENON_PARAMS).segment === nothing
    seg = evolve(on, y0, HENON_PARAMS; trajectory = true).segment
    @test seg isa TrajectorySegment
    @test seg.kind == :discrete && seg.mode == :H && seg.t === nothing
    @test size(seg.y) == (2, 2)
    @test seg.y[:, 1] ≈ y0
    @test seg.y[:, 2] ≈ henon(y0, HENON_PARAMS)
    proj = DiscreteMode(:P, reduce_u; in_trajectory = true)
    res = evolve(proj, [1.0, 2.0]; trajectory = true)
    @test res.segment === nothing
    @test res.y == [2.0]
end

@testitem "mode construction errors" setup = [Fixtures] begin
    @test_throws MethodError DiscreteMode(:bad, (y, p) -> y)
    @test_throws MethodError ContinuousMode(:bad, (y, p) -> y)
    @test_throws MethodError ContinuousMode(:bad, henon)
    @test_throws MethodError DiscreteMode(:bad, izhikevich)
    @test_throws ArgumentError ContinuousMode(:A, izhikevich; borders = (izhikevich,))
    b3 = Border((y, _) -> y[3], 3)
    @test_throws ArgumentError ContinuousMode(:A, izhikevich; borders = (b3,))
    @test_throws ArgumentError ContinuousMode(:A, izhikevich; borders = (fire, fire))
    single = ContinuousMode(:A, izhikevich; borders = fire)
    @test border_names(single) == (:fire,)
    @test occursin("A", repr(single))
end

@testitem "continuous mode hits the firing border" setup = [Fixtures] begin
    mode = ContinuousMode(:A, izhikevich; borders = (fire,))
    res = evolve(mode, IZH_Y0_MODE, IZH_PARAMS)
    @test res.border == 1
    @test res.t > 0
    @test res.y[1] ≈ 30.0 atol = 1.0e-6
    @test res.jac === nothing && res.hes === nothing && res.segment === nothing

    full = evolve(mode, IZH_Y0_MODE, IZH_PARAMS; hessian = true)
    @test size(full.jac) == (2, 2)
    @test size(full.hes) == (2, 2, 2)
    # The image lies on the border v = 30, so the first row vanishes.
    @test all(abs.(full.jac[1, :]) .< 1.0e-12)
    @test all(abs.(full.hes[1, :, :]) .< 1.0e-12)
    # The Hessian is symmetric in its last two indices.
    @test full.hes ≈ permutedims(full.hes, (1, 3, 2)) atol = 1.0e-12
    @test full.y ≈ res.y rtol = 1.0e-6
    @test full.t ≈ res.t rtol = 1.0e-6
end

@testitem "continuous mode derivatives match finite differences and the reference" setup = [Fixtures] begin
    mode = ContinuousMode(:A, izhikevich; borders = (fire,), TIGHT...)
    res = evolve(mode, IZH_Y0_MODE, IZH_PARAMS; hessian = true)
    f(y) = evolve(mode, y, IZH_PARAMS).y
    @test res.jac ≈ fd_jacobian(f, IZH_Y0_MODE, 1.0e-4) atol = 1.0e-6
    @test maximum(abs, res.hes) > 1.0e-2
    @test res.hes ≈ fd_hessian(f, IZH_Y0_MODE, 1.0e-4) atol = 1.0e-4

    ref = reference["izhikevich"]["mode_step"]
    @test refvec(ref["y0"]) == IZH_Y0_MODE
    @test res.y ≈ refvec(ref["y"]) rtol = 1.0e-7
    @test res.t ≈ ref["t"] rtol = 1.0e-7
    @test res.border == ref["border"] + 1
    @test res.jac ≈ refmat(ref["jac"]) rtol = 1.0e-6
    @test res.hes ≈ reften(ref["hes"]) rtol = 1.0e-5
end

@testitem "parametrised border with a non-trivial gradient" setup = [Fixtures] begin
    using StaticArrays: SVector
    rot = VectorField((y, p) -> SVector(-p.w * y[2] + 0.5 * y[1], p.w * y[1]), 2)
    circle = Border((y, p) -> y[1]^2 + y[2]^2 - p.r^2, 2)
    mode = ContinuousMode(:rot, rot; borders = (circle,), TIGHT...)
    params = (w = 1.0, r = 1.5)
    y0 = [0.6, -0.3]
    res = evolve(mode, y0, params; hessian = true)
    @test res.border == 1
    @test sqrt(sum(abs2, res.y)) ≈ 1.5 atol = 1.0e-8
    f(y) = evolve(mode, y, params).y
    @test res.jac ≈ fd_jacobian(f, y0) atol = 1.0e-6
    @test res.hes ≈ fd_hessian(f, y0) atol = 1.0e-4

    ref = reference["rotation_mode_step"]
    @test res.y ≈ refvec(ref["y"]) rtol = 1.0e-7
    @test res.t ≈ ref["t"] rtol = 1.0e-7
    @test res.jac ≈ refmat(ref["jac"]) rtol = 1.0e-6
    @test res.hes ≈ reften(ref["hes"]) rtol = 1.0e-5
end

@testitem "continuous trajectory segment" setup = [Fixtures] begin
    mode = ContinuousMode(:A, izhikevich; borders = (fire,))
    res = evolve(mode, IZH_Y0_MODE, IZH_PARAMS; trajectory = true)
    seg = res.segment
    @test seg isa TrajectorySegment
    @test seg.kind == :continuous && seg.mode == :A
    @test seg.t[1] == 0.0
    @test seg.t[end] ≈ res.t
    @test size(seg.y, 1) == 2
    @test size(seg.y, 2) == length(seg.t)
    @test dim(seg) == 2
    @test all(diff(seg.t) .> 0)
    @test seg.y[:, 1] ≈ IZH_Y0_MODE
    @test seg.y[:, end] ≈ res.y rtol = 1.0e-8
    quiet = ContinuousMode(:A, izhikevich; borders = (fire,), in_trajectory = false)
    @test evolve(quiet, IZH_Y0_MODE, IZH_PARAMS; trajectory = true).segment === nothing
    @test_throws ArgumentError evolve(mode, [1.0, 2.0, 3.0], IZH_PARAMS)
    @test_throws ArgumentError evolve(mode, 1.0, IZH_PARAMS)
end

@testitem "mode not terminated" begin
    using StaticArrays: SVector
    far = Border((y, _) -> y[1] - 1.0e6, 2)
    slow = VectorField((y, _) -> SVector(one(y[1]), zero(y[1])), 2)
    mode = ContinuousMode(:slow, slow; borders = (far,), max_interval = 1.0)
    err = try
        evolve(mode, [0.0, 0.0])
        nothing
    catch e
        e
    end
    @test err isa ModeNotTerminatedError
    @test err isa MappyError
    @test err.mode == :slow
    @test err.max_interval == 1.0
    @test occursin("slow", sprint(showerror, err))
end

@testitem "border direction is respected" begin
    using StaticArrays: SVector
    up = VectorField((y, _) -> SVector(one(y[1])), 1)
    at_one = Border((y, _) -> y[1] - 1, 1; direction = -1)
    at_two = Border((y, _) -> y[1] - 2, 1; direction = 1)
    mode = ContinuousMode(:up, up; borders = (at_one, at_two))
    res = evolve(mode, [0.0])
    @test res.border == 2
    @test res.y[1] ≈ 2.0 atol = 1.0e-8
    @test res.t ≈ 2.0 atol = 1.0e-8
    both = ContinuousMode(:up, up; borders = (Border((y, _) -> y[1] - 1, 1; direction = 0), at_two))
    @test evolve(both, [0.0]).border == 1
end

@testitem "borderless mode is the flow map" setup = [Fixtures] begin
    using LinearAlgebra
    A = [0.1 1.0; -2.0 -0.3]
    linear = VectorField((y, _) -> A * y, 2)
    t_end = 1.5
    mode = ContinuousMode(:lin, linear; max_interval = t_end, TIGHT...)
    y0 = [1.0, 0.5]
    res = evolve(mode, y0; hessian = true)
    @test res.border === nothing
    @test res.t ≈ t_end
    flow = exp(A * t_end)
    @test res.y ≈ flow * y0 rtol = 1.0e-8
    @test res.jac ≈ flow rtol = 1.0e-8
    @test all(abs.(res.hes) .< 1.0e-10)
    @test res.jac ≈ fd_jacobian(y -> evolve(mode, y).y, y0) atol = 1.0e-6
    default = ContinuousMode(:lin, linear; max_interval = 1.0)
    @test evolve(default, y0; jacobian = true).jac ≈ exp(A) atol = 1.0e-5
end

@testitem "discrete mode parameter Jacobian" setup = [Fixtures] begin
    mode = DiscreteMode(:B, firing)
    params = (c = -50.0, d = 2.0, unused = 1.0)
    res = evolve(mode, [30.0, -1.0], params; parameter_jacobian = true)
    @test res.jac_p !== nothing && size(res.jac_p) == (2, 3)
    @test res.jac_p ≈ [1.0 0.0 0.0; 0.0 1.0 0.0]
    @test evolve(mode, [30.0, -1.0], params).jac_p === nothing
    @test_throws ArgumentError evolve(mode, [30.0, -1.0], Dict(:c => -50.0, :d => 2.0); parameter_jacobian = true)
end

@testitem "continuous mode parameter Jacobian matches finite differences" setup = [Fixtures] begin
    mode = ContinuousMode(:A, izhikevich; borders = (fire,), TIGHT...)
    res = evolve(mode, IZH_Y0_MODE, IZH_PARAMS; parameter_jacobian = true)
    @test res.jac_p !== nothing && size(res.jac_p) == (2, 5)
    @test res.jac !== nothing  # implied
    h = 1.0e-6
    for (m, key) in enumerate(keys(IZH_PARAMS))
        up = evolve(mode, IZH_Y0_MODE, merge(IZH_PARAMS, NamedTuple{(key,)}((IZH_PARAMS[key] + h,)))).y
        dn = evolve(mode, IZH_Y0_MODE, merge(IZH_PARAMS, NamedTuple{(key,)}((IZH_PARAMS[key] - h,)))).y
        @test res.jac_p[:, m] ≈ (up - dn) / (2h) rtol = 1.0e-5 atol = 1.0e-6
    end
end

@testitem "parameter Jacobian with a parametrised border" setup = [Fixtures] begin
    # ẏ = (1, p y1) stopped at y1 = c: T(y) = (c, y2 + p (c² - y1²) / 2).
    using StaticArrays: SVector
    field = VectorField((y, p) -> SVector(one(y[1]), p.p * y[1]), 2)
    stop = Border((y, p) -> y[1] - p.c, 2)
    mode = ContinuousMode(:flow, field; borders = (stop,), TIGHT...)
    params = (p = 0.7, c = 2.0)
    y0 = [0.5, -1.0]
    res = evolve(mode, y0, params; parameter_jacobian = true)
    c, p = params.c, params.p
    @test res.y ≈ [c, y0[2] + p * (c^2 - y0[1]^2) / 2] rtol = 1.0e-8
    @test res.jac_p ≈ [0.0 1.0; (c^2 - y0[1]^2) / 2 p * c] rtol = 1.0e-7 atol = 1.0e-9
    # The hitting time depends on c only through the border: without the
    # correction term the derivative with respect to c would vanish.
    @test res.jac_p[1, 2] ≈ 1.0 rtol = 1.0e-8
end
