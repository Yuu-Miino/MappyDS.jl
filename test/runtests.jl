using MappyDS
using Test
using Aqua
using JET
using ExplicitImports
using ReTestItems

@testset "MappyDS.jl" begin
    # Unit tests are `@testitem`s in test/*_test.jl (one file per Python test
    # module: functions, modes, maps, continuation, root, package). The
    # cross-check against the Python implementation (test/crosscheck_test.jl,
    # PythonCall + mappy-ds) runs only when MAPPYDS_CROSSCHECK=1.
    @testset "Unit tests (ReTestItems)" begin
        runtests(MappyDS)
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MappyDS)
    end
    @testset "Code linting (JET.jl)" begin
        # Only errors located in MappyDS itself are reported: the abstract
        # analysis of the SciML and DifferentiationInterface internals that
        # the package calls produces many false positives otherwise.
        # `target_defined_modules` exists up to JET 0.9 (the series for Julia
        # 1.10); later versions analyse the package's own modules by default.
        jet_kwargs = pkgversion(JET) < v"0.10" ? (; target_defined_modules = true) : (;)
        JET.test_package(MappyDS; toplevel_logger = nothing, target_modules = (MappyDS,), jet_kwargs...)
    end
    @testset "Explicit imports" begin
        @test check_no_implicit_imports(MappyDS) === nothing
        @test check_no_stale_explicit_imports(MappyDS) === nothing
    end
end
