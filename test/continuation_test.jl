@testitem "natural continuation sweeps linearly" begin
    sqrt_step(y, p) = StepFunctionResult(true, sqrt(p.p))
    steps = continuation(sqrt_step, 1.0, (p = 1.0,), :p, 4.0; resolution = 7)
    @test length(steps) == 7
    values = [s.params.p for s in steps]
    @test values ≈ collect(range(1.0, 4.0; length = 7))
    @test values[1] == 1.0 && values[end] == 4.0
    @test all(s isa ContinuationStep for s in steps)
    @test all(isempty(s.info) for s in steps)
    @test [s.y for s in steps] ≈ sqrt.(values)
    down = continuation(sqrt_step, 2.0, (p = 4.0,), :p, 1.0; resolution = 4)
    @test [s.params.p for s in down] == [4.0, 3.0, 2.0, 1.0]
end

@testitem "continuation seeds, stops and carries parameters" begin
    seen = Int[]
    step(y, p) = (push!(seen, y); StepFunctionResult(true, y + 1))
    steps = continuation(step, 0, (p = 0.0,), :p, 1.0; resolution = 4)
    @test seen == [0, 1, 2, 3]
    @test [s.y for s in steps] == [1, 2, 3, 4]

    calls = Float64[]
    failing(y, p) = (push!(calls, p.p); StepFunctionResult(p.p < 2.5, y))
    steps = continuation(failing, 0.0, (p = 0.0,), :p, 5.0; resolution = 6)
    @test [s.params.p for s in steps] == [0.0, 1.0, 2.0]
    @test calls == [0.0, 1.0, 2.0, 3.0]

    none(y, p) = StepFunctionResult(true, p.p >= 1.0 ? nothing : y)
    @test [s.params.p for s in continuation(none, 0.0, (p = 0.0,), :p, 2.0; resolution = 5)] == [0.0, 0.5]
    @test isempty(continuation((y, p) -> StepFunctionResult(false), 0.0, (p = 0.0,), :p, 1.0))

    received = NamedTuple[]
    function solved(y, p)
        push!(received, p)
        q = merge(p, (q = p.q + 10.0,))
        return StepFunctionResult(true, y, q, Dict{Symbol, Any}(:q => q.q))
    end
    steps = continuation(solved, 0.0, (p = 0.0, q = 0.0), :p, 2.0; resolution = 3)
    @test [r.q for r in received] == [0.0, 10.0, 20.0]
    @test [s.params.q for s in steps] == [10.0, 20.0, 30.0]
    @test [r.p for r in received] == [0.0, 1.0, 2.0]
    @test [s.info[:q] for s in steps] == [10.0, 20.0, 30.0]

    collected = ContinuationStep[]
    cb(y, p) = StepFunctionResult(p.p < 3.5, p.p)
    steps = continuation(cb, 0.0, (p = 0.0,), :p, 5.0; resolution = 6, callback = s -> push!(collected, s))
    @test length(collected) == length(steps) == 4
    @test collected == steps
    for r in (1, 0, -3)
        @test_throws ArgumentError continuation(cb, 1.0, (p = 1.0,), :p, 2.0; resolution = r)
    end
    @test_throws ArgumentError continuation(cb, 1.0, (p = 1.0,), :q, 2.0)
end

@testitem "continuation progress output" begin
    sqrt_step(y, p) = StepFunctionResult(true, sqrt(p.p))
    text = mktemp() do path, io
        redirect_stderr(io) do
            continuation(sqrt_step, 1.0, (p = 1.0,), :p, 4.0; resolution = 4, show_progress = true)
        end
        flush(io)
        return read(path, String)
    end
    @test count("SUCCESS", text) == 4
    @test occursin("4/4", text)
    @test occursin("p=+4", text)
    @test endswith(text, "\n")

    boom(y, p) = p.p > 1.5 ? throw(DivideError()) : StepFunctionResult(true, y)
    text = mktemp() do path, io
        run() = redirect_stderr(() -> continuation(boom, 0.0, (p = 1.0,), :p, 2.0; resolution = 3, show_progress = true), io)
        @test_throws DivideError run()
        flush(io)
        return read(path, String)
    end
    @test endswith(text, "\n")

    quiet = mktemp() do path, io
        redirect_stderr(io) do
            continuation(sqrt_step, 1.0, (p = 1.0,), :p, 4.0; resolution = 4)
        end
        flush(io)
        return read(path, String)
    end
    @test quiet == ""
end

@testitem "arclength continuation on the unit circle" begin
    using LinearAlgebra
    circle(u) = ([dot(u, u) - 1.0], permutedims(2u))
    pts = arclength_continuation(circle, [1.0, 0.0]; step = 0.3, step_max = 0.3)
    @test 15 < length(pts) < 40
    for p in pts
        @test abs(dot(p.u, p.u) - 1.0) < 1.0e-9
        @test abs(norm(p.tangent) - 1.0) < 1.0e-12
        @test abs(dot(p.tangent, p.u)) < 1.0e-8
    end
    @test pts[2].u[2] > 0  # direction = +1: the last component increases first
    signs = [sign(p.tangent[2]) for p in pts]
    @test count(signs[2:end] .!= signs[1:(end - 1)]) == 2  # two folds
    @test norm(pts[end].u - pts[1].u) < 0.5  # closed on itself

    down = arclength_continuation(circle, [1.0, 0.0]; step = 0.1, step_max = 0.1, direction = -1, bounds = (-0.5, 1.0))
    @test down[2].u[2] < 0
    @test all(p.u[2] >= -0.5 for p in down)
    @test down[end].u[2] < -0.3

    start = arclength_continuation(circle, [0.7, 0.3]; step = 0.1, max_steps = 0)
    @test length(start) == 1
    @test start[1].u[2] ≈ 0.3
    @test start[1].u[1] ≈ sqrt(1 - 0.09)
    @test start[1].step == 0.0

    seen = ArclengthPoint[]
    cb = arclength_continuation(circle, [1.0, 0.0]; step = 0.3, step_max = 0.3, callback = p -> push!(seen, p))
    @test length(seen) == length(cb)
    @test all(a === b for (a, b) in zip(seen, cb))

    @test_throws ArgumentError arclength_continuation(circle, [1.0]; step = 0.1)
    @test_throws ArgumentError arclength_continuation(circle, [1.0, 0.0]; direction = 2)
    @test_throws ArgumentError arclength_continuation(circle, [1.0, 0.0]; step = 1.0, step_max = 0.1)
    nowhere(u) = ([dot(u, u) + 1.0], permutedims(2u))
    @test_throws "starting point" arclength_continuation(nowhere, [1.0, 0.0])
end

@testitem "arclength continuation passes a fold and retries failed steps" begin
    using LinearAlgebra
    # Fixed points of y -> y + p - y²: the curve p = y² folds at (0, 0).
    system(u) = ([u[2] - u[1] * u[1]], [-2.0 * u[1] 1.0])
    pts = arclength_continuation(system, [0.5, 0.25]; step = 0.05, step_max = 0.1, direction = -1, bounds = (-1, 0.3))
    ys = [p.u[1] for p in pts]
    ps = [p.u[2] for p in pts]
    @test ps ≈ ys .^ 2 atol = 1.0e-9
    @test minimum(ps) < 0.01
    @test ys[end] < -0.3 && ps[end] > 0.1

    circle(u) = ([dot(u, u) - 1.0], permutedims(2u))
    calls = Ref(0)
    function flaky(u)
        calls[] += 1
        if u[2] > 0.6 && calls[] % 7 == 0
            throw(SingularException(1))
        end
        return circle(u)
    end
    fpts = arclength_continuation(flaky, [1.0, 0.0]; step = 0.05, step_min = 1.0e-4, step_max = 0.4, bounds = (-2, 0.9))
    steps = [p.step for p in fpts[2:end]]
    @test maximum(steps) > 0.05
    @test minimum(steps) > 0
    for p in fpts
        @test abs(dot(p.u, p.u) - 1.0) < 1.0e-9
    end
end
