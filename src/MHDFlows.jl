module MHDFlows

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

include("pgen.jl")
include("MHDSolver.jl")
include("MHDSolver_VP.jl")
include("HDSolver.jl")
include("datastructure.jl")
include("integrator.jl")
include("pgenCho.jl")
include("pgenAlfven.jl")
include("pgenVP.jl")
include("VectorCalculus.jl")
include("MHDAnalysis.jl")

@reexport using MHDFlows.pgen
@reexport using MHDFlows.MHDSolver
@reexport using MHDFlows.MHDSolver_VP
@reexport using MHDFlows.HDSolver
@reexport using MHDFlows.datastructure
@reexport using MHDFlows.integrator
@reexport using MHDFlows.pgenCho
@reexport using MHDFlows.pgenAlfven
@reexport using MHDFlows.pgenVP
@reexport using MHDFlows.VectorCalculus
@reexport using MHDFlows.MHDAnalysis


end