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
