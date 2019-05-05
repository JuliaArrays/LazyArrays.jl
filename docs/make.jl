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

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
