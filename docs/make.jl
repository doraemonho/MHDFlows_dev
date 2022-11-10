using Documenter
using MHDFlows

makedocs(
    sitename = "MHDFlows",
    format = Documenter.HTML(),
    modules = [MHDFlows]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
