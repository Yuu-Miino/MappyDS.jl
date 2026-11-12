# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
as interpreted by Julia's Pkg (in `0.x`, a minor bump is breaking).

## [Unreleased]

## [0.3.1] - 2026-11-12

Documentation and project metadata only; no change to the package code.

### Added

- `bench/`: per-operation benchmarks of MappyDS.jl against `mappy-ds`
  (Python) on the same systems, the manual *Benchmark* workflow that runs
  both on one runner, and a "Performance" page in the docs with the
  measured table.
- `CONTRIBUTING.md` with the development workflow and the release
  procedure; `.zenodo.json` and `CITATION.cff` with the author metadata; the
  Zenodo concept DOI (`10.5281/zenodo.24152370`) in the README and
  `CITATION.bib`; a coverage badge.

### Fixed

- The worked examples show their printed results before the figures.
- TagBot pushes tags with a deploy key of this repository (`TAGBOT_KEY`), so
  that the versioned docs are built for each tag.

## [0.3.0] - 2026-11-09

Mixed derivatives and local bifurcation analysis (the features of
`mappy-ds` 0.3.0).

### Added

- Mixed second derivatives: `mixed_hessian(f, y, p)` for system functions,
  `evolve(...; parameter_hessian = true)` returning `StepResult.hes_p`
  (`∂²y_i/∂y0_j∂p_m`) from a third variational system in continuous modes
  with its border correction (including borders that depend on
  parameters), and `image_detail(...; parameter_hessian = true)` returning
  `MapResult.hes_p` through the chain rule across modes.
- `find_local_bf`: tangent, period-doubling and Neimark–Sacker bifurcation
  points of periodic points, solved from `(Tⁿ(y) - y, Re det A, Im det A)`
  with `A = DTⁿ - e^{iθ} I` and the fully analytic Jacobian of the system
  (cofactor matrix; `jac`, `hes`, `jac_p` and `hes_p` of the map);
  `LocalBfResult`.
- `trace_local_bf` (natural-parameter continuation of a bifurcation point
  along a second parameter) and `trace_local_bf_arclength` (pseudo-arclength
  continuation of the bifurcation curve in the `(y, λ_bf, θ, λ_cnt)` space,
  passing cusps and flagging `:fold`, the strong resonances
  `:resonance_1_1` … `:resonance_1_4` on Neimark–Sacker curves and
  `:second_multiplier_crossing`).
- `examples/`: the five Literate.jl scripts of the Python package (Hénon,
  three coupled logistic maps, Izhikevich, forced Duffing, forced Alpazur),
  rendered with their output in the "Working examples" section of the docs.
- Plotting through the package extension `MappyDSMakieExt` (loaded with a
  Makie backend): `plot_trajectory` draws a `Trajectory` onto an `Axis`, and
  `plot2d(map, y0, p, config::PlotConfig)` iterates a map and draws its orbit,
  interactively (GLMakie: click to restart, `f` toggles trajectories,
  space clears, `p` and up/down change a parameter) or for a fixed number of
  `frames`; `PlotConfig` and `LineStyle` configure it.

## [0.2.0] - 2026-11-09

Parameter derivatives of the Poincaré map and pseudo-arclength
continuation (the features of `mappy-ds` 0.2.0).

### Added

- Derivatives with respect to the parameters: `parameter_jacobian(f, y, p)`
  for system functions (ForwardDiff over the values of the `NamedTuple`
  `p`), `evolve(...; parameter_jacobian = true)` returning
  `StepResult.jac_p` from the sensitivity equations of continuous modes with
  the border correction (including borders that depend on parameters), and
  `image_detail(...; parameter_jacobian = true)` returning
  `MapResult.jac_p` (columns ordered as `MapResult.param_keys`) through the
  chain rule across modes.
- Pseudo-arclength continuation: `arclength_continuation` (generic
  predictor–corrector on a bordered system, adaptive step, fold passing,
  closed-curve detection) returning `ArclengthPoint`s, and
  `trace_cycle_arclength`, which follows a branch of periodic points through
  folds using `jac` and `jac_p` for the tangent and flags `:fold`,
  `:branch_point`, `:period_doubling` and `:neimark_sacker` events from sign
  changes between consecutive points.

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

[Unreleased]: https://github.com/Yuu-Miino/MappyDS.jl/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/Yuu-Miino/MappyDS.jl/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Yuu-Miino/MappyDS.jl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Yuu-Miino/MappyDS.jl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Yuu-Miino/MappyDS.jl/releases/tag/v0.1.0
