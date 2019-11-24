using Documenter
using LazyArrays

makedocs(
    sitename = "LazyArrays",
    modules = [LazyArrays],
    pages = [
        "Home" => "index.md",
        hide("internals.md"),
    ],
)

deploydocs(
    repo = "github.com/JuliaArrays/LazyArrays.jl"
)
