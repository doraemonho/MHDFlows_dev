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
include("Solver/HDSolver.jl")
include("Solver/MHDSolver.jl")
include("Solver/HDSolver_VP.jl")
include("Solver/MHDSolver_VP.jl")
include("DiagnosticWrapper.jl")
include("integrator.jl")
include("datastructure.jl")
include("utils/VectorCalculus.jl")
include("utils/MHDAnalysis.jl")
include("utils/GeometryFunction.jl")
include("utils/func.jl")

export Problem,             # Simulation Related function
       TimeIntegrator!,
       Restart!,
       Cylindrical_Mask_Function,
       SetUpProblemIC!
       Curl,                # Vector Calculas Related function
       Div,
       LaplaceSolver,
       Crossproduct,
       Dotproduct,
       xy_to_polar,         # Geometry Related function
       ScaleDecomposition,  # MHD Analysis Related function 
       h_k,
       VectorPotential,
       LaplaceSolver,
       getL

end