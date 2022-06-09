module pgenVP2
#Problem Gernerator for setting Up the problem 

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

@reexport using FourierFlows



include("MHDSolver_VP.jl")
include("datastructure.jl")

export VP_Problem2,DivBCorrection

MHDcalcN_advection!  = MHDSolver_VP.MHDcalcN_advection!;

nothingfunction(args...) = nothing;

function VP_Problem2(dev::Device=CPU();
              # Numerical parameters
                            nx = 64,
                            ny = nx,
                            nz = nx,
                            Lx = 2π,
                            Ly = Lx,
                            Lz = Lx,
               # Drag and/or hyper-viscosity for velocity/B-field
                             ν = 0,
                            nν = 1,
                             η = 0,
              # Timestepper and equation options
                            dt = 0.01,
                       stepper = "RK4",
              # Force Driving parameters       
                         calcF = nothingfunction,
              # Maskfunction 
                   Maskfuncion = nothingfunction,
              # Float type and dealiasing
                             T = Float32)

  VP_Params = dev == GPU() ? VP_Params_GPU : VP_Params_CPU;

  # set up the grid function   
  grid = ThreeDGrid(dev, nx, Lx, ny, Ly, nz, Lz; T=T)
  
  # set up the mask function
  χ = Maskfuncion(grid);
  χ = dev == GPU() ? CuArray(χ) : χ;
  U₀x = dev == GPU() ? CUDA.zeros(T,size(χ)) : zeros(T,size(χ));
  U₀y = dev == GPU() ? CUDA.zeros(T,size(χ)) : zeros(T,size(χ));
  U₀z = dev == GPU() ? CUDA.zeros(T,size(χ)) : zeros(T,size(χ));
  B₀x = dev == GPU() ? CUDA.zeros(T,size(χ)) : zeros(T,size(χ));
  B₀y = dev == GPU() ? CUDA.zeros(T,size(χ)) : zeros(T,size(χ));
  B₀z = dev == GPU() ? CUDA.zeros(T,size(χ)) : zeros(T,size(χ));

  if size(χ)[1] != nx || size(χ)[2] != ny || size(χ)[3] != nz
    error("The Shape of Mask doesn't match the grid size!")
  end  
        
  # set up the function for this three
  params = VP_Params{T}(ν, η, nν, 1, 2, 3, 4, 5, 6, calcF, χ, U₀x, U₀y, U₀z, B₀x, B₀y, B₀z)

  vars = SetVP_Vars(dev, grid);

  equation = Equation_with_forcing(dev, params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)

end

function DivBCorrection(prob)
#= 
   Possion Solver for periodic boundary condition
   As in VP method, ∇ ⋅ B = 0 doesn't hold, B_{t+1} = ∇×Ψ + ∇Φ -> ∇ ⋅ B = ∇² Φ
   We need to find Φ and remove it using a Poission Solver 
   Here we are using the Fourier Method to find the Φ
   In Real Space,  
   ∇² Φ = ∇ ⋅ B   
   In k-Space,  
   ∑ᵢ -(kᵢ)² Φₖ = i∑ᵢ kᵢ(Bₖ)ᵢ
   Φ = F{ i∑ᵢ kᵢ (Bₖ)ᵢ / ∑ᵢ (k²)ᵢ}
=#  

	vars = prob.vars;
	grid = prob.grid;
  params = prob.params;
    #find Φₖ
    kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
    k⁻² = grid.invKrsq;
    @. vars.nonlin1  *= 0;
    @. vars.nonlinh1 *= 0;       
    ∑ᵢkᵢBᵢh_k² = vars.nonlinh1;
    ∑ᵢkᵢBᵢ_k²  = vars.nonlin1;
    bxh = prob.sol[:, :, :, params.bx_ind];
    byh = prob.sol[:, :, :, params.by_ind];
    bzh = prob.sol[:, :, :, params.bz_ind];
    ∑ᵢkᵢBᵢh_k² = @. -im*(kᵢ*bxh + kⱼ*byh + kₖ*bzh);
    ∑ᵢkᵢBᵢh_k² = @. ∑ᵢkᵢBᵢh_k²*k⁻²;  # Φₖ
    
    # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
    bxh  .-= kᵢ.*∑ᵢkᵢBᵢh_k²;
    byh  .-= kⱼ.*∑ᵢkᵢBᵢh_k²;
    bzh  .-= kₖ.*∑ᵢkᵢBᵢh_k²;
    
    #Update to Real Space vars
    ldiv!(vars.bx, grid.rfftplan, deepcopy(bxh));# deepcopy() since inverse real-fft destroys its input
    ldiv!(vars.by, grid.rfftplan, deepcopy(byh));# deepcopy() since inverse real-fft destroys its input
    ldiv!(vars.bz, grid.rfftplan, deepcopy(bzh));# deepcopy() since inverse real-fft destroys its input
end

function DivVCorrection(prob)
#= 
   Possion Solver for periodic boundary condition
   As in VP method, ∇ ⋅ B = 0 doesn't hold, B_{t+1} = ∇×Ψ + ∇Φ -> ∇ ⋅ B = ∇² Φ
   We need to find Φ and remove it using a Poission Solver 
   Here we are using the Fourier Method to find the Φ
   In Real Space,  
   ∇² Φ = ∇ ⋅ B   
   In k-Space,  
   ∑ᵢ -(kᵢ)² Φₖ = i∑ᵢ kᵢ(Bₖ)ᵢ
   Φ = F{ i∑ᵢ kᵢ (Bₖ)ᵢ / ∑ᵢ (k²)ᵢ}
=#  

  vars = prob.vars;
  grid = prob.grid;
  params = prob.params;
    #find Φₖ
    kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
    k⁻² = grid.invKrsq;
    @. vars.nonlin1  *= 0;
    @. vars.nonlinh1 *= 0;       
    ∑ᵢkᵢUᵢh_k² = vars.nonlinh1;
    ∑ᵢkᵢUᵢ_k²  = vars.nonlin1;
    uxh = prob.sol[:, :, :, params.ux_ind];
    uyh = prob.sol[:, :, :, params.uy_ind];
    uzh = prob.sol[:, :, :, params.uz_ind];
    ∑ᵢkᵢUᵢh_k² = @. -im*(kᵢ*uxh + kⱼ*uyh + kₖ*uzh);
    ∑ᵢkᵢUᵢh_k² = @. ∑ᵢkᵢUᵢh_k²*k⁻²;  # Φₖ
    
    # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
    uxh  .-= kᵢ.*∑ᵢkᵢUᵢh_k²;
    uyh  .-= kⱼ.*∑ᵢkᵢUᵢh_k²;
    uzh  .-= kₖ.*∑ᵢkᵢUᵢh_k²;
    
    #Update to Real Space vars
    ldiv!(vars.ux, grid.rfftplan, deepcopy(uxh));# deepcopy() since inverse real-fft destroys its input
    ldiv!(vars.uy, grid.rfftplan, deepcopy(uyh));# deepcopy() since inverse real-fft destroys its input
    ldiv!(vars.uz, grid.rfftplan, deepcopy(uzh));# deepcopy() since inverse real-fft destroys its input
end



abstract type MHDVars <: AbstractVars end
struct VP_Vars{Aphys, Atrans} <: MHDVars
    "x-component of velocity"
        ux :: Aphys
    "y-component of velocity"
        uy :: Aphys
    "z-component of velocity"
        uz :: Aphys
    "x-component of B-field"
        bx :: Aphys
    "y-component of B-field"
        by :: Aphys
    "z-component of B-field"
        bz :: Aphys
    "Fourier transform of x-component of velocity"
       uxh :: Atrans
    "Fourier transform of y-component of velocity"
       uyh :: Atrans
    "Fourier transform of z-component of velocity"
       uzh :: Atrans
    "Fourier transform of x-component of B-field"
       bxh :: Atrans
    "Fourier transform of y-component of B-field"
       byh :: Atrans
    "Fourier transform of z-component of B-field"
       bzh :: Atrans

    # Temperatory Cache 
    "Non-linear term 1"
     nonlin1 :: Aphys
    "Fourier transform of Non-linear term"
    nonlinh1 :: Atrans

    # Forcing vars
    "x-component of force"
      fx :: Aphys
    "y-component of force"
      fy :: Aphys
    "z-component of force"
      fz :: Aphys
    "k_f*θ"
      kfθ :: Aphys
end

struct VP_Params_CPU{T} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
    ν :: T
  "small-scale (hyper)-viscosity coefficient for b"
    η :: T
  "(hyper)-viscosity order, `nν```≥ 1``"
    nν :: Int

  "Array Indexing for velocity"
    ux_ind :: Int
    uy_ind :: Int
    uz_ind :: Int
    
  "Array Indexing for B-field"
    bx_ind :: Int
    by_ind :: Int
    bz_ind :: Int
  "function that calculates the Fourier transform of the forcing, ``F̂``"
    calcF! :: Function 

  "Volume penzlization method paramter"
    χ   :: Array{T}
    U₀x :: Array{T}
    U₀y :: Array{T}
    U₀z :: Array{T}
    B₀x :: Array{T}
    B₀y :: Array{T}
    B₀z :: Array{T}
end

struct VP_Params_GPU{T} <: AbstractParams
    
  "small-scale (hyper)-viscosity coefficient for v"
    ν :: T
  "small-scale (hyper)-viscosity coefficient for b"
    η :: T
  "(hyper)-viscosity order, `nν```≥ 1``"
    nν :: Int

  "Array Indexing for velocity"
    ux_ind :: Int
    uy_ind :: Int
    uz_ind :: Int
    
  "Array Indexing for B-field"
    bx_ind :: Int
    by_ind :: Int
    bz_ind :: Int

  "function that calculates the Fourier transform of the forcing, ``F̂``"
    calcF! :: Function 

  "Volume penzlization method paramter"
    χ   :: CuArray{T}
    U₀x :: CuArray{T}
    U₀y :: CuArray{T}
    U₀z :: CuArray{T}
    B₀x :: CuArray{T}
    B₀y :: CuArray{T}
    B₀z :: CuArray{T}
end

function SetVP_Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid) 
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1 fx fy fz kfθ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) uxh uyh uzh bxh byh bzh nonlinh1 
    
  VP_Vars( ux,  uy,  uz,  bx,  by,  bz,
          uxh, uyh, uzh, bxh, byh, bzh,
          nonlin1, nonlinh1,
          fx, fy, fz, kfθ);
end

function MHDcalcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

function addforcing!(N, sol, t, clock, vars, params, grid)
  params.calcF!(N, sol, t, clock, vars, params, grid) 
  return nothing

end

function Equation_with_forcing(dev,params::VP_Params_CPU, grid::AbstractGrid)
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr, grid.nl, grid.nm, 6));

  return FourierFlows.Equation(L,MHDcalcN!, grid)
end

function Equation_with_forcing(dev,params::VP_Params_GPU, grid::AbstractGrid)
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr, grid.nl, grid.nm, 6));

  return FourierFlows.Equation(L,MHDcalcN!, grid)
end

end