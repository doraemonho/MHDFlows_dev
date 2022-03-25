module MHDFlows

using 
  CUDA,
  Statisctics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

include("pgen.jl")
include("Solver.jl")
include("datastructure.jl")

@reexport using MHDFlows.pgen
@reexport using MHDFlows.solver
@reexport using MHDFlows.datastructure

end