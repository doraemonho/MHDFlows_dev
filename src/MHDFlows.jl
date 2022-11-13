module MHDFlows

using 
  CUDA,
  Statistics,
  Reexport,
  DocStringExtensions,
  HDF5,
  FFTW,
  ProgressMeter

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
import Base: show, summary

"Abstract supertype for problem."
abstract type AbstractProblem end
abstract type MHDVars <: AbstractVars end

include("DyeModule.jl")
include("Problems.jl")
include("pgen.jl")
include("Solver/VPSolver.jl")
include("Solver/HDSolver.jl")
include("Solver/MHDSolver.jl")
include("Solver/HDSolver_Compessible.jl")
include("Solver/MHDSolver_Compessible.jl")
include("DiagnosticWrapper.jl")
include("integrator.jl")
include("datastructure.jl")
include("utils/utils.jl");
include("utils/VectorCalculus.jl")
include("utils/MHDAnalysis.jl")
include("utils/GeometryFunction.jl")
include("utils/IC.jl")
include("utils/UserInterface.jl")

#pgen module
include("pgen/A99ForceDriving.jl")
include("pgen/TaylorGreenDynamo.jl")
include("pgen/NegativeDamping.jl")

export Problem,           
       TimeIntegrator!,
       Restart!,
       Cylindrical_Mask_Function,
       DivFreeSpectraMap,
       SetUpProblemIC!,
       readMHDFlows,
       Curl,            
       Div,
       LaplaceSolver,
       Crossproduct,
       Dotproduct,
       ∂i,∇X,
       xy_to_polar,       
       ScaleDecomposition, 
       h_k,
       h_m,
       VectorPotential,
       LaplaceSolver,
       getL,
       spectralline,
       ⋅, ×
end