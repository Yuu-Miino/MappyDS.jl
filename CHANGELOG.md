# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
as interpreted by Julia's Pkg (in `0.x`, a minor bump is breaking).

## [Unreleased]

## [0.1.0] - 2026-11-08

Port of `mappy-ds` 0.1.0 (the mode concept and the numerics of Miino,
*NOLTA, IEICE* 15(2), 2024) to Julia.

### Added

- System functions `VectorField(f, dim)`, `Mapping(f, dom_dim, cod_dim)` and
  `Border(q, dim; direction, name)` wrapping a user function `f(y, p)`;
  `jacobian` and `hessian` of a system function are computed with
  ForwardDiff on `SVector`s (the Python package traces the function with
  SymPy instead), so branches on the state are allowed.
- `ContinuousMode(name, fun; borders, max_interval, alg, reltol, abstol)`
  (OrdinaryDiffEq with one `ContinuousCallback` per border) and
  `DiscreteMode(name, fun)`; `evolve(mode, y0, p; jacobian, hessian,
  trajectory)` returns a `StepResult`. The Jacobian and the Hessian of a
  continuous mode come from the variational equations integrated with the
  state and the analytic correction for the state-dependent hitting time.
- `PoincareMap(modes, transitions, initial_mode, end_modes; max_transitions)`
  with the three forms of transition rules (a mode name, one name per
  border, `Dict(:border => :mode)`) and validation of the mode graph at
  construction; `Diffeomorphism(fun)` for a single discrete map; `image`,
  `image_detail` (chain rule across modes, `mode` override) and
  `trajectory`; `MapResult` with `LinearAlgebra.eigvals` / `eigvecs`.
- `Trajectory` / `TrajectorySegment` containers.
- `continuation` (natural-parameter sweep with a step function),
  `ContinuationStep`, `StepFunctionResult`.
- `find_cycle` (Newton / trust region through SimpleNonlinearSolve with the
  analytic Jacobian) returning `CycleResult`, and `trace_cycle`.
- Parameters are passed through to the user functions as given (a
  `NamedTuple` is recommended); the keys of a `NamedTuple` are reported in
  `MapResult.param_keys`.
- Exceptions `MappyError`, `UnknownModeError`, `TransitionError`,
  `ModeNotTerminatedError`, `DerivativeError`; plotting stubs for the Makie
  extension (MappyDS 0.3).
- Test suite mirroring the Python tests (Izhikevich neuron, Hénon map,
  forced Duffing oscillator, relay oscillator with a closed-form Poincaré
  map; finite-difference checks) and comparing against reference values
  exported from `mappy-ds` 0.3.0 (`test/fixtures/reference.json`).
- CI (tests on Julia LTS / stable / pre-release, Runic, docs), TagBot,
  Dependabot, Documenter deployment into `doc.mappy/julia/`.

[Unreleased]: https://github.com/Yuu-Miino/MappyDS.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Yuu-Miino/MappyDS.jl/releases/tag/v0.1.0
