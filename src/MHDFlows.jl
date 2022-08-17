module MHDFlows

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions,
  HDF5

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
import Base: show, summary

"Abstract supertype for problem."
abstract type AbstractProblem end
abstract type MHDVars <: AbstractVars end

include("DyeModule.jl")
include("Problems.jl")
include("pgen.jl")
include("HDSolver.jl")
include("MHDSolver.jl")
include("HDSolver_VP.jl")
include("MHDSolver_VP.jl")
include("DiagnosticWrapper.jl")
include("integrator.jl")
include("datastructure.jl")
include("utils/VectorCalculus.jl")
include("utils/MHDAnalysis.jl")
include("utils/GeometryFunction.jl")
include("utils/func.jl")

export Problem,
       TimeIntegrator!,
       Restart!,
       Curl,
       Div,
       LaplaceSolver,
       Crossproduct,
       Dotproduct,
       xy_to_polar,
       ScaleDecomposition,
       h_k,
       VectorPotential,
       LaplaceSolver,
       getL

end