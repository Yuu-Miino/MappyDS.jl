@testitem "Izhikevich paper values" setup = [Fixtures] begin
    paper = reference["izhikevich"]["paper"]["I10"]
    pmap = make_izhikevich_map()
    res = image_detail(pmap, -3.0, IZH_PARAMS; hessian = true)
    @test size(res.y) == (1,) && size(res.jac) == (1, 1) && size(res.hes) == (1, 1, 1)
    @test res.y[1] ≈ paper["y"] rtol = 1.0e-5
    @test res.jac[1, 1] ≈ paper["jac"] rtol = 1.0e-5
    @test res.hes[1, 1, 1] ≈ paper["hes"] rtol = 1.0e-5
    @test res.mode_history == [:Pinv, :B, :A, :P, :Pinv]
    @test end_mode(res) == :Pinv
    @test res.param_keys == (:a, :b, :c, :d, :I)
    @test dim(pmap) == 1
    res3 = image_detail(pmap, -3.0, IZH_PARAMS; n = 3, jacobian = false)
    @test length(res3.mode_history) == 1 + 3 * 4
    @test res3.mode_history[1:5] == res.mode_history
    @test count(==(:A), res3.mode_history) == 3
end

@testitem "Izhikevich matches the converged reference" setup = [Fixtures] begin
    pmap = make_izhikevich_map(; TIGHT...)
    for case in reference["izhikevich"]["image"]
        p = refparams(case["params"])
        n = length(case["mode_history"]) ÷ 4
        res = image_detail(pmap, refvec(case["y0"]), p; n, hessian = true)
        @test res.y ≈ refvec(case["y"]) rtol = 1.0e-7
        @test res.jac ≈ refmat(case["jac"]) rtol = 1.0e-6
        @test res.hes ≈ reften(case["hes"]) rtol = 1.0e-5
        @test res.mode_history == Symbol.(case["mode_history"])
    end
    paper = reference["izhikevich"]["paper"]["I10"]
    res = image_detail(pmap, -3.0, IZH_PARAMS; hessian = true)
    # Against the published values: y agrees to ~1.0e-8, jac/hes to ~1.0e-6/1.0e-5.
    @test res.y[1] ≈ paper["y"] rtol = 1.0e-7
    @test res.jac[1, 1] ≈ paper["jac"] rtol = 2.0e-6
    @test res.hes[1, 1, 1] ≈ paper["hes"] rtol = 2.0e-5
    f(u) = image(pmap, u, IZH_PARAMS)
    @test res.jac ≈ fd_jacobian(f, [-3.0]) atol = 1.0e-6
    @test res.hes ≈ fd_hessian(f, [-3.0]) atol = 1.0e-4
end

@testitem "image, image_detail flags and iterates" setup = [Fixtures] begin
    pmap = make_izhikevich_map()
    y = image(pmap, -3.0, IZH_PARAMS; n = 2)
    detail = image_detail(pmap, -3.0, IZH_PARAMS; n = 2, jacobian = false)
    @test y == detail.y
    with_jac = image_detail(pmap, -3.0, IZH_PARAMS; n = 2)
    @test y ≈ with_jac.y rtol = 1.0e-5
    y1 = image(pmap, -3.0, IZH_PARAMS)
    @test y ≈ image(pmap, y1, IZH_PARAMS) rtol = 1.0e-6
    res = image_detail(pmap, -3.0, IZH_PARAMS)
    @test res.jac !== nothing && res.hes === nothing && res.trajectory === nothing
    res = image_detail(pmap, -3.0, IZH_PARAMS; jacobian = false)
    @test res.jac === nothing && res.hes === nothing
    @test_throws ArgumentError image(pmap, -3.0, IZH_PARAMS; n = 0)
    @test_throws ArgumentError image(pmap, -3.0, IZH_PARAMS; n = -1)
    @test_throws ArgumentError image_detail(pmap, [1.0, 2.0], IZH_PARAMS)
end

@testitem "MapResult eigenvalues" setup = [Fixtures] begin
    using LinearAlgebra
    hmap = henon_map()
    res = image_detail(hmap, [0.1, 0.2], HENON_PARAMS; n = 2)
    vals = eigvals(res)
    vecs = eigvecs(res)
    @test vals isa Vector{ComplexF64} && vecs isa Matrix{ComplexF64}
    @test sort(real.(vals)) ≈ sort(real.(eigvals(res.jac)))
    for (lam, v) in zip(vals, eachcol(vecs))
        @test res.jac * v ≈ lam * v atol = 1.0e-12
    end
    dmap = make_duffing_map()
    dres = image_detail(dmap, DUFFING_Y0, DUFFING_PARAMS)
    dvals = eigvals(dres)
    @test all(abs.(imag.(dvals)) .> 0)  # a stable focus
    @test all(abs.(dvals) .< 1)
    plain = image_detail(hmap, [0.1, 0.2], HENON_PARAMS; jacobian = false)
    @test_throws ArgumentError eigvals(plain)
    @test_throws ArgumentError eigvecs(plain)
    @test_throws ArgumentError eigvals(MapResult(zeros(2); jac = ones(2, 3)))
end

@testitem "Hénon diffeomorphism" setup = [Fixtures] begin
    using LinearAlgebra: eigvals
    hmap = henon_map()
    @test hmap.fun === henon
    @test dim(hmap) == 2
    @test occursin("henon", repr(hmap))
    y0 = [0.1, -0.2]
    manual = y0
    for _ in 1:3
        manual = henon(manual, HENON_PARAMS)
    end
    @test image(hmap, y0, HENON_PARAMS; n = 3) ≈ manual rtol = 1.0e-14
    res = image_detail(hmap, y0, HENON_PARAMS; n = 3, hessian = true)
    f(y) = image(hmap, y, HENON_PARAMS; n = 3)
    @test res.jac ≈ fd_jacobian(f, y0, 1.0e-5) atol = 1.0e-8
    @test res.hes ≈ fd_hessian(f, y0, 1.0e-4) atol = 1.0e-6
    y1 = henon(y0, HENON_PARAMS)
    y2 = henon(y1, HENON_PARAMS)
    chain = jacobian(henon, y2, HENON_PARAMS) * jacobian(henon, y1, HENON_PARAMS) * jacobian(henon, y0, HENON_PARAMS)
    @test res.jac ≈ chain rtol = 1.0e-13

    ref = reference["henon"]["image"][1]
    rres = image_detail(hmap, refvec(ref["y0"]), refparams(ref["params"]); n = 3, hessian = true)
    @test rres.y ≈ refvec(ref["y"]) rtol = 1.0e-12
    @test rres.jac ≈ refmat(ref["jac"]) rtol = 1.0e-12
    @test rres.hes ≈ reften(ref["hes"]) rtol = 1.0e-12

    fp = reference["henon"]["fixed_point"]
    fres = image_detail(hmap, refvec(fp["y"]), HENON_PARAMS)
    @test fres.y ≈ refvec(fp["y"]) atol = 1.0e-10
    @test sort(real.(eigvals(fres))) ≈ sort(real.(refcomplex(fp["eigvals"]))) rtol = 1.0e-8

    tres = image_detail(hmap, [0.0, 0.0], HENON_PARAMS; n = 3, trajectory = true)
    @test tres.mode_history == fill(:map, 4)
    traj = tres.trajectory
    @test length(traj) == 3
    @test all(s.kind == :discrete for s in traj)
    @test size(traj.y) == (2, 6)
    @test traj[end].y[:, end] ≈ tres.y
    named = Diffeomorphism(henon; name = :H)
    @test image_detail(named, [0.0, 0.0], HENON_PARAMS).mode_history == [:H, :H]
    proj = Mapping((y, _) -> y[1], 2, 1)
    @test_throws ArgumentError Diffeomorphism(proj)
    traj2 = trajectory(hmap, y0, HENON_PARAMS; n = 2)
    @test length(traj2) == 2
    @test traj2.y ≈ hcat(y0, y1, y1, y2)
end

@testitem "Duffing period-one point" setup = [Fixtures] begin
    using LinearAlgebra: eigvals
    dmap = make_duffing_map()
    @test dim(dmap) == 2
    res = image_detail(dmap, DUFFING_Y0, DUFFING_PARAMS)
    @test res.mode_history == [:p, :duffing, :t_reset, :pinv, :p]
    @test size(res.y) == (2,) && size(res.jac) == (2, 2)
    @test res.y ≈ DUFFING_Y0 atol = 1.0e-5

    tight = make_duffing_map(; TIGHT...)
    tres = image_detail(tight, DUFFING_Y0, DUFFING_PARAMS; hessian = true)
    f(y) = image(tight, y, DUFFING_PARAMS)
    @test tres.jac ≈ fd_jacobian(f, DUFFING_Y0) atol = 1.0e-6
    @test tres.hes ≈ fd_hessian(f, DUFFING_Y0) atol = 1.0e-4

    ref = reference["duffing"]["image"][1]
    @test tres.y ≈ refvec(ref["y"]) rtol = 1.0e-7
    @test tres.jac ≈ refmat(ref["jac"]) rtol = 1.0e-6
    @test tres.hes ≈ reften(ref["hes"]) rtol = 1.0e-5
    fp = reference["duffing"]["fixed_point"]
    fres = image_detail(tight, refvec(fp["y"]), DUFFING_PARAMS)
    @test fres.y ≈ refvec(fp["y"]) atol = 1.0e-8
    @test sort(eigvals(fres); by = imag) ≈ sort(refcomplex(fp["eigvals"]); by = imag) rtol = 1.0e-6
end

@testitem "relay oscillator: two end modes and mode override" setup = [Fixtures] begin
    rmap = make_relay_map()
    res = image_detail(rmap, [0.0], RELAY_PARAMS; hessian = true)
    @test res.mode_history == [:p_up, :up, :down, :r_down, :pinv_down, :p_down]
    @test end_mode(res) == :p_down
    @test res.y ≈ [-0.2] atol = 1.0e-9
    @test res.jac ≈ [-1.0;;] atol = 1.0e-9
    @test all(abs.(res.hes) .< 1.0e-9)

    over = image_detail(rmap, [-0.2], RELAY_PARAMS; mode = :p_down)
    @test over.mode_history[1] == :p_down
    @test end_mode(over) == :p_up
    @test over.y ≈ [0.4] atol = 1.0e-9
    @test over.jac ≈ [-1.0;;] atol = 1.0e-9
    @test image(rmap, [-0.2], RELAY_PARAMS; mode = :p_down) ≈ over.y

    two = image_detail(rmap, [0.0], RELAY_PARAMS; n = 2)
    @test end_mode(two) == :p_up
    @test count(==(:p_down), two.mode_history) == 1
    @test two.y ≈ [0.4] atol = 1.0e-9
    @test two.jac ≈ [1.0;;] atol = 1.0e-9
    @test_throws UnknownModeError image_detail(rmap, [0.0], RELAY_PARAMS; mode = :nope)

    for case in reference["relay"]["image"]
        n = count(==(:up) ∘ Symbol, case["mode_history"]) + count(==(:down) ∘ Symbol, case["mode_history"])
        kw = case["mode_history"][1] == "p_down" ? (; mode = :p_down) : (;)
        r = image_detail(rmap, refvec(case["y0"]), RELAY_PARAMS; n = n ÷ 2, hessian = true, kw...)
        @test r.y ≈ refvec(case["y"]) atol = 1.0e-9
        @test r.jac ≈ refmat(case["jac"]) atol = 1.0e-9
        @test r.mode_history == Symbol.(case["mode_history"])
    end
    text = repr(rmap)
    @test occursin("p_up", text) && occursin("p_down", text)
end

@testitem "trajectories" setup = [Fixtures] begin
    pmap = make_izhikevich_map()
    traj = trajectory(pmap, -3.0, IZH_PARAMS; n = 2)
    @test traj isa Trajectory
    @test length(traj) == 2
    @test [s.mode for s in traj] == [:A, :A]
    @test all(s.kind == :continuous for s in traj)
    y = traj.y
    @test size(y) == (2, sum(size(s.y, 2) for s in traj))
    @test y[:, 1:size(traj[1].y, 2)] ≈ traj[1].y
    for seg in traj
        @test seg.y[1, 1] ≈ -50.0
        @test seg.y[1, end] ≈ 30.0 atol = 1.0e-6
    end
    res = image_detail(pmap, -3.0, IZH_PARAMS; trajectory = true)
    @test res.trajectory isa Trajectory
    @test length(res.trajectory) == 1
    @test res.trajectory[1].y[2, end] ≈ res.y[1]

    mixed = Trajectory(
        [
            TrajectorySegment(:a, :discrete, zeros(2, 2)),
            TrajectorySegment(:b, :continuous, zeros(3, 4), 0.0:3.0),
        ]
    )
    @test [dim(s) for s in mixed] == [2, 3]
    @test_throws ArgumentError mixed.y
    s1 = TrajectorySegment(:a, :continuous, ones(2, 3), [0.0, 1.0, 2.0])
    s2 = TrajectorySegment(:b, :discrete, zeros(2, 2))
    @test Trajectory([s1, s2]).y == hcat(ones(2, 3), zeros(2, 2))
    @test states(Trajectory([s1, s2])) == Trajectory([s1, s2]).y
    @test size(Trajectory().y) == (0, 0)
    @test length(Trajectory()) == 0
    @test_throws ArgumentError TrajectorySegment(:a, :weird, zeros(2, 2))
    @test_throws DimensionMismatch TrajectorySegment(:a, :continuous, zeros(2, 2), [0.0])
end

@testitem "PoincareMap construction errors" setup = [Fixtures] begin
    ident = Mapping((y, _) -> y, 1; name = :ident)
    ab = [DiscreteMode(:a, ident), DiscreteMode(:b, ident)]
    @test_throws ArgumentError PoincareMap([DiscreteMode(:a, ident), DiscreteMode(:a, ident)], Dict(:a => :a), :a)
    for (transitions, initial, ends) in (
            (Dict(:x => :a), :a, nothing),
            (Dict(:a => :x), :a, nothing),
            (Dict(:a => [:x]), :a, nothing),
            (Dict(:a => :b), :x, nothing),
            (Dict(:a => :b), :a, :x),
            (Dict(:a => :b), :a, [:b, :x]),
        )
        err = try
            PoincareMap(ab, transitions, initial, ends)
            nothing
        catch e
            e
        end
        @test err isa UnknownModeError
        @test err.name == :x
        @test occursin(":x", sprint(showerror, err))
    end
    err = try
        PoincareMap(ab, Dict(:a => :b), :a)
    catch e
        e
    end
    @test err isa TransitionError && err.mode == :b
    @test occursin("reachable", sprint(showerror, err))
    pm = PoincareMap([ab..., DiscreteMode(:c, ident)], Dict(:a => :a), :a)
    @test image(pm, [1.0]) == [1.0]

    modes = [ContinuousMode(:A, izhikevich; borders = (fire,)), DiscreteMode(:B, firing)]
    @test_throws "expected 1" PoincareMap(modes, Dict(:B => :A, :A => Symbol[]), :B)
    fire2 = Border((y, _) -> y[1] - 40.0, 2; name = :fire2)
    @test_throws "single successor" PoincareMap([ContinuousMode(:A, izhikevich; borders = (fire, fire2)), modes[2]], Dict(:B => :A, :A => :B), :B)

    named = [ContinuousMode(:A, izhikevich; borders = (fire, fire2)), DiscreteMode(:B, firing), DiscreteMode(:C, firing)]
    by_name = PoincareMap(named, Dict(:B => :A, :C => :A, :A => Dict(:fire2 => :C, :fire => :B)), :B)
    by_list = PoincareMap(named, Dict(:B => :A, :C => :A, :A => [:B, :C]), :B)
    @test by_name.transitions == by_list.transitions == Dict(:B => (:A,), :C => (:A,), :A => (:B, :C))
    @test image_detail(by_name, [-50.0, -1.7], IZH_PARAMS).mode_history == [:B, :A, :B]
    @test_throws "no successor for border(s) [:fire2]" PoincareMap(named, Dict(:B => :A, :C => :A, :A => Dict(:fire => :B)), :B)
    @test_throws "unknown border(s) [:nope]" PoincareMap(named, Dict(:B => :A, :C => :A, :A => Dict(:fire => :B, :fire2 => :C, :nope => :B)), :B)
    @test_throws "no borders" PoincareMap(ab, Dict(:a => Dict(:x => :b), :b => :a), :a)
    proj = Mapping((y, _) -> y[1], 2, 1)
    @test_throws "dimension 1 but :p starts in dimension 2" PoincareMap([DiscreteMode(:p, proj)], Dict(:p => :p), :p)
    @test_throws "no end mode" PoincareMap([ab..., DiscreteMode(:c, ident)], Dict(:a => :b, :b => :a), :a, :c)
    decay = VectorField((y, _) -> -y, 1)
    @test_throws "expected 1" PoincareMap([ContinuousMode(:flow, decay; max_interval = 1.0)], Dict(:flow => [:flow, :flow]), :flow)
end

@testitem "borderless flow and the transition bound" setup = [Fixtures] begin
    decay = VectorField((y, _) -> -y, 1)
    pmap = PoincareMap([ContinuousMode(:flow, decay; max_interval = 1.0, TIGHT...)], Dict(:flow => :flow), :flow)
    res = image_detail(pmap, [2.0]; n = 2)
    @test res.y ≈ [2.0 * exp(-2.0)] rtol = 1.0e-8
    @test res.jac ≈ [exp(-2.0);;] rtol = 1.0e-8

    grow = VectorField((y, _) -> y, 1)
    near = Border((y, _) -> y[1] - 2.0, 1; name = :near)
    far = Border((y, _) -> y[1] - 1.0e6, 1; name = :far)
    halve = Mapping((y, _) -> y / 2, 1)
    ident = Mapping((y, _) -> y, 1)
    modes = [
        ContinuousMode(:flow, grow; borders = (near, far), max_interval = 100.0),
        DiscreteMode(:reset, halve),
        DiscreteMode(:end, ident),
    ]
    loop = PoincareMap(modes, Dict(:flow => Dict(:near => :reset, :far => :end), :reset => :flow, :end => :end), :flow, :end; max_transitions = 10)
    @test loop.max_transitions == 10
    @test_throws "10 transitions" image(loop, [1.0])
end
