using MappyDS
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Literate

DocMeta.setdocmeta!(MappyDS, :DocTestSetup, :(using MappyDS); recursive = true)

# The worked examples are Literate.jl scripts in examples/; they are converted
# to Markdown pages with executable `@example` blocks (CairoMakie figures).
const EXAMPLES = ["henon", "logistic_3d", "izhikevich", "duffing", "forced_alpazur"]
const EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")
mkpath(EXAMPLE_DIR)
for name in EXAMPLES
    Literate.markdown(joinpath(@__DIR__, "..", "examples", "$name.jl"), EXAMPLE_DIR; documenter = true)
end

# Cross-references to the Python documentation (Sphinx objects.inv), e.g.
# [`PoincareMap`](@extref mappy mappy_ds.PoincareMap).
links = InterLinks(
    "mappy" => "https://yuu-miino.github.io/doc.mappy/latest/objects.inv",
)
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

makedocs(;
    modules = [MappyDS],
    authors = "Yuu Miino",
    sitename = "MappyDS.jl",
    format = Documenter.HTML(;
        canonical = "https://yuu-miino.github.io/doc.mappy/julia/stable/",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "User guide" => [
            "guide/modes.md",
            "guide/derivatives.md",
            "guide/root.md",
            "guide/plotting.md",
        ],
        "Working examples" => ["examples.md"; ["examples/$name.md" for name in EXAMPLES]],
        "API reference" => "api.md",
        "Python package" => "python.md",
        "References" => "references.md",
    ],
    plugins = [links, bib],
)

# The docs live in the shared documentation repository `doc.mappy`, next to
# the Sphinx docs of the Python package, under the subdirectory `julia/`.
# Pushing to another repository requires the SSH deploy key DOCUMENTER_KEY
# (GITHUB_TOKEN only reaches the repository the workflow runs in).
deploydocs(;
    repo = "github.com/Yuu-Miino/MappyDS.jl.git",
    deploy_repo = "github.com/Yuu-Miino/doc.mappy.git",
    branch = "gh-pages",
    dirname = "julia",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
    push_preview = false,
)
