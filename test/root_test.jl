@testitem "find_cycle: Hénon fixed point and 2-cycle" setup = [Fixtures] begin
    using LinearAlgebra: eigvals
    hmap = henon_map()
    res = find_cycle(hmap, [0.6, 0.2], HENON_PARAMS)
    @test res isa CycleResult
    @test res.success
    @test size(res.y) == (2,)
    @test res.y ≈ [0.631354477, 0.189406343] rtol = 1.0e-8
    @test sort(real.(res.eigvals)) ≈ [-1.923738858, 0.155946322] rtol = 1.0e-8
    @test size(res.eigvecs) == (2, 2)
    @test res.nfev > 0
    @test all(abs.(res.residual) .< 1.0e-9)
    ref = reference["henon"]["fixed_point"]
    @test res.y ≈ refvec(ref["y"]) rtol = 1.0e-9

    two = find_cycle(hmap, [1.0, -0.1], HENON_PARAMS; period = 2)
    @test two.success
    y = two.y
    @test image(hmap, y, HENON_PARAMS; n = 2) ≈ y atol = 1.0e-10
    @test sqrt(sum(abs2, image(hmap, y, HENON_PARAMS) - y)) > 0.1
    @test y ≈ [0.97580005, -0.14274002] rtol = 1.0e-7
    @test y ≈ refvec(reference["henon"]["period_two"]["y"]) rtol = 1.0e-9
    jac2 = image_detail(hmap, y, HENON_PARAMS; n = 2).jac
    @test sort(two.eigvals; by = real) ≈ sort(ComplexF64.(eigvals(jac2)); by = real)

    attractor = image(hmap, [0.1, 0.1], HENON_PARAMS; n = 1002)
    from_attractor = find_cycle(hmap, attractor, HENON_PARAMS; period = 2)
    @test from_attractor.success
    @test image(hmap, from_attractor.y, HENON_PARAMS; n = 2) ≈ from_attractor.y atol = 1.0e-10
    @test_throws ArgumentError find_cycle(hmap, [0.6, 0.2], HENON_PARAMS; period = 0)
end

@testitem "find_cycle: Izhikevich fixed point" setup = [Fixtures] begin
    pmap = make_izhikevich_map()
    p8 = merge(IZH_PARAMS, (I = 8.0,))
    res = find_cycle(pmap, [-4.6], p8)
    @test res.success
    @test res.y ≈ [-4.645418] rtol = 1.0e-5
    @test real(res.eigvals[1]) ≈ 0.221961 rtol = 1.0e-4
    paper = reference["izhikevich"]["paper"]["I8"]
    tight = make_izhikevich_map(; TIGHT...)
    tres = find_cycle(tight, [-4.6], p8)
    @test tres.success
    @test tres.y[1] ≈ paper["u"] rtol = 1.0e-6
    detail = image_detail(tight, tres.y, p8; hessian = true)
    @test detail.jac[1, 1] ≈ paper["jac"] rtol = 1.0e-7
    @test detail.hes[1, 1, 1] ≈ paper["hes"] rtol = 1.0e-7
    ref = reference["izhikevich"]["fixed_point_I8"]
    @test tres.y ≈ refvec(ref["y"]) rtol = 1.0e-8
    @test detail.jac ≈ refmat(ref["jac"]) rtol = 1.0e-6
    @test detail.hes ≈ reften(ref["hes"]) rtol = 1.0e-5
end

@testitem "find_cycle failures" setup = [Fixtures] begin
    no_fixed_point = Mapping((x, _) -> x + 1 + x^2, 1)
    res = find_cycle(Diffeomorphism(no_fixed_point), [0.0])
    @test !res.success
    @test res.y === nothing && res.eigvals === nothing && res.eigvecs === nothing
    @test size(res.residual) == (1,)
    limited = find_cycle(henon_map(), [0.0, 0.0], HENON_PARAMS; maxiters = 2)
    @test !limited.success
    @test limited.y === nothing
end

@testitem "trace_cycle" setup = [Fixtures] begin
    pmap = make_izhikevich_map()
    p8 = merge(IZH_PARAMS, (I = 8.0,))
    start = find_cycle(pmap, [-4.6], p8)
    @test start.success
    steps = trace_cycle(pmap, start.y, p8, :I, 4.5; resolution = 10)
    @test length(steps) == 10
    currents = [s.params.I for s in steps]
    @test currents[1] == 8.0 && currents[end] == 4.5
    @test all(diff(currents) .< 0)
    for s in steps
        @test issubset([:eigvals, :eigvecs, :nfev, :residual], keys(s.info))
        @test size(s.y) == (1,)
    end
    mu = [real(s.info[:eigvals][1]) for s in steps]
    @test mu[1] ≈ 0.221961 rtol = 1.0e-4
    @test all(diff(mu) .< 0)
    @test mu[end] < -1  # period doubling crossed

    seen = ContinuationStep[]
    hmap = henon_map()
    hsteps, text = mktemp() do path, io
        steps = redirect_stderr(io) do
            trace_cycle(hmap, [0.631354477, 0.189406343], HENON_PARAMS, :a, 1.2; resolution = 3, callback = s -> push!(seen, s), show_progress = true)
        end
        flush(io)
        return steps, read(path, String)
    end
    @test occursin("SUCCESS", text)
    @test seen == hsteps
    @test length(hsteps) == 3
    for s in hsteps
        a, b = s.params.a, s.params.b
        x = (-(1 - b) + sqrt((1 - b)^2 + 4a)) / (2a)
        @test s.y ≈ [x, b * x] rtol = 1.0e-8
    end

    saddle_node = Mapping((x, p) -> x + p.p - x^2, 1)
    sn = trace_cycle(Diffeomorphism(saddle_node), [0.5], (p = 0.25,), :p, -0.25; resolution = 3)
    @test [s.params.p for s in sn] == [0.25, 0.0]
    @test sn[1].y ≈ [0.5]
    @test sn[2].y ≈ [0.0] atol = 1.0e-4
end
