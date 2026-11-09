# Contributing

MappyDS.jl follows [ColPrac](https://github.com/SciML/ColPrac). Bug reports,
questions and pull requests are welcome on the
[issue tracker](https://github.com/Yuu-Miino/MappyDS.jl/issues).

## Development

```julia
using Pkg
Pkg.develop(path = ".")      # from a clone of the repository
Pkg.test("MappyDS")          # unit tests (ReTestItems), Aqua, JET, ExplicitImports
```

- Tests live in `test/*_test.jl` as `@testitem`s; shared systems and the
  reference values exported from the Python package
  (`test/fixtures/reference.json`, produced by `tests/export_reference.py`
  in the `mappy` repository) are in `test/fixtures_testsetup.jl`.
- Code is formatted with [Runic](https://github.com/fredrikekre/Runic.jl)
  (`runic --inplace src test ext examples docs`); CI checks it.
- Imports are explicit (`using X: a, b`), including in the Makie extension;
  ExplicitImports checks it.
- The docs are built with Documenter from `docs/`; the worked examples in
  `examples/` are Literate.jl scripts converted at build time:

  ```julia
  using Pkg; Pkg.activate("docs"); Pkg.develop(path = "."); Pkg.instantiate()
  include("docs/make.jl")    # output in docs/build/
  ```

- CI runs on pushes and pull requests to `main`. For another branch, start it
  by hand from the Actions tab (`CI` → *Run workflow*).

## Releasing

Releases follow [Semantic Versioning](https://semver.org/) as interpreted by
Pkg: in `0.x`, a minor bump is breaking. Versions are registered in the
[General registry](https://github.com/JuliaRegistries/General).

1. Bump `version` in `Project.toml` (patch, minor or major +1 from the last
   registered version; AutoMerge rejects anything else).
2. Move the `[Unreleased]` section of `CHANGELOG.md` to
   `## [X.Y.Z] - YYYY-MM-DD` and update the links at the end of the file.
   Mark breaking changes explicitly.
3. Merge into `main` with CI green, then comment on that commit:

   ```
   @JuliaRegistrator register

   Release notes:

   <the CHANGELOG section of this version>
   ```

4. Registrator opens a pull request in General. AutoMerge checks the name,
   `[compat]` upper bounds, the license and that the package installs and
   loads; a new package waits three days, a new version of an existing
   package about fifteen minutes.
5. After the merge, TagBot creates the tag `vX.Y.Z` and a GitHub release
   (with the release notes). The tag push rebuilds the docs into
   `doc.mappy/julia/vX.Y/` and updates `stable`.

The docs are deployed to the shared documentation repository
[`doc.mappy`](https://github.com/Yuu-Miino/doc.mappy) under `julia/`, next to
the Sphinx docs of the Python package. This needs the secret `DOCUMENTER_KEY`
(an SSH deploy key of `doc.mappy` with write access, base64-encoded) in this
repository; TagBot uses the same key.
