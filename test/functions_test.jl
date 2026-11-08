@testitem "system function attributes" setup = [Fixtures] begin
    @test izhikevich isa VectorField
    @test (dom_dim(izhikevich), cod_dim(izhikevich)) == (2, 2)
    @test nameof(izhikevich) == :izhikevich
    @test occursin("izhikevich", repr(izhikevich))

    @test henon isa Mapping
    @test (dom_dim(henon), cod_dim(henon)) == (2, 2)
    @test (dom_dim(reduce_u), cod_dim(reduce_u)) == (2, 1)
    @test (dom_dim(revert), cod_dim(revert)) == (1, 2)
    @test occursin("dom_dim = 2", repr(reduce_u))

    @test fire isa Border
    @test (dom_dim(fire), cod_dim(fire)) == (2, 1)
    @test fire.direction == 1
    for d in (-1, 0, 1)
        b = Border((y, _) -> y[1], 1; direction = d)
        @test b.direction == d
    end
    @test_throws ArgumentError Border((y, _) -> y[1], 1; direction = 2)
    @test_throws ArgumentError VectorField((y, _) -> y, 0)
    @test_throws ArgumentError Mapping((y, _) -> y, 2, 0)
end

@testitem "evaluation returns Float64 vectors" setup = [Fixtures] begin
    y = henon([0.5, 0.1], HENON_PARAMS)
    @test y isa Vector{Float64}
    @test y ≈ [1 - 1.4 * 0.25 + 0.1, 0.15]
    @test henon((0.5, 0.1), HENON_PARAMS) == y
    @test henon([1, 0], HENON_PARAMS) == [1 - 1.4, 0.3]
    # Scalars are accepted for one-dimensional domains, scalar outputs for
    # one-dimensional codomains.
    @test revert(-3.0, nothing) == [30.0, -3.0]
    @test reduce_u([30.0, -3.0]) == [-3.0]
    @test_throws ArgumentError henon([1.0, 2.0, 3.0], HENON_PARAMS)
    @test_throws ArgumentError henon([1.0 2.0; 3.0 4.0], HENON_PARAMS)
    # Wrong number of components returned by the user function.
    bad = Mapping((y, _) -> SVector(y[1], y[2], 0.0), 2, 2)
    @test_throws DimensionMismatch bad([1.0, 2.0])
end

@testitem "Hénon Jacobian and Hessian" setup = [Fixtures] begin
    y0 = [0.5, 0.1]
    jac = jacobian(henon, y0, HENON_PARAMS)
    @test jac isa Matrix{Float64}
    @test jac ≈ [-1.4 1.0; 0.3 0.0]
    hes = hessian(henon, y0, HENON_PARAMS)
    @test hes isa Array{Float64, 3}
    @test size(hes) == (2, 2, 2)
    @test hes[1, 1, 1] ≈ -2.8
    @test all(iszero, hes[:, :, 2])
    @test all(iszero, hes[2, :, :])
    @test jac ≈ fd_jacobian(y -> henon(y, HENON_PARAMS), y0, 1.0e-5) atol = 1.0e-8
    @test hes ≈ fd_hessian(y -> henon(y, HENON_PARAMS), y0) atol = 1.0e-6
end

@testitem "derivatives of a non-square map and of a border" setup = [Fixtures] begin
    @test jacobian(reduce_u, [1.0, 2.0]) == [0.0 1.0]
    @test jacobian(revert, [2.0]) == reshape([0.0, 1.0], 2, 1)
    @test size(hessian(revert, [2.0])) == (2, 1, 1)
    @test jacobian(fire, [30.0, -1.0]) == [1.0 0.0]
    @test hessian(fire, [30.0, -1.0]) == zeros(1, 2, 2)
    # Izhikevich Jacobian at the origin, as in the Python docstring.
    @test jacobian(izhikevich, [0.0, 0.0], (a = 0.2, b = 0.2, I = 10.0)) ≈ [5.0 -1.0; 0.04 -0.2]
end

@testitem "a function may be shared by modes and branch on the state" setup = [Fixtures] begin
    f = Mapping((y, _) -> y[1] > 0 ? SVector(2y[1], y[2]) : SVector(y[1], 3y[2]), 2)
    @test f([1.0, 1.0]) == [2.0, 1.0]
    @test f([-1.0, 1.0]) == [-1.0, 3.0]
    @test jacobian(f, [1.0, 1.0]) == [2.0 0.0; 0.0 1.0]
    @test jacobian(f, [-1.0, 1.0]) == [1.0 0.0; 0.0 3.0]
    m1 = DiscreteMode(:a, henon)
    m2 = DiscreteMode(:b, henon)
    @test m1.fun === m2.fun
    @test evolve(m1, [0.5, 0.1], HENON_PARAMS).y == evolve(m2, [0.5, 0.1], HENON_PARAMS).y
end

@testitem "as_state" begin
    @test as_state(1) == [1.0]
    @test as_state((1, 2)) == [1.0, 2.0]
    v = [1.0, 2.0]
    w = as_state(v)
    @test w == v && w !== v
    @test_throws ArgumentError as_state(ones(2, 2))
    @test_throws ArgumentError as_state("x")
end
