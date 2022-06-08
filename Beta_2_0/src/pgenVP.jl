module pgenVP
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

export VP_Problem,DivBCorrection

MHDcalcN_advection!  = MHDSolver_VP.MHDcalcN_advection!;

nothingfunction(args...) = nothing;

function VP_Problem(dev::Device=CPU();
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
                            A0 = 1.0,
                            kf = 1.0,
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
  fo,s1,s2 = Setupfos1s2(grid;kf=kf);
  params = VP_Params{T}(ν, η, nν, 1, 2, 3, 4, 5, 6, calcF, fo, s1, s2, χ, U₀x, U₀y, U₀z, B₀x, B₀y, B₀z)

  vars = SetVP_Vars(dev, grid);
  setupChovars!(vars;A0=A0);

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
struct VP_Vars{Aphys, Atrans, Avars1, Avars2} <: MHDVars
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
    "x-component of Additional Force term"
          Fx :: Aphys
    "y-component of Additional Force term"
          Fy :: Aphys
    "x-component of Additional Force term"
         Fxh :: Atrans
    "y-component of Additional Force term"
         Fyh :: Atrans
    # Forcing vars
    "Forcing Amplitude"
     A  :: Avars1
    "Random Phase 1"
     Φ1 :: Avars2
    "Random Phase 2"
     Φ2 :: Avars2
     "Ω"
      Ω :: Avars1
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

  "vector for forcing"
    fo :: Array{Int32}
    s1 :: Array{T}
    s2 :: Array{T}

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

  "vector for forcing"
    fo :: Array{Int32}
    s1 :: Array{T}
    s2 :: Array{T}

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
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1 Fx Fy
  @devzeros Dev T (22) Φ1 Φ2
  @devzeros Dev T (1) A Ω
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) uxh uyh uzh bxh byh bzh nonlinh1 Fgx Fgy
  #χ = Dev == typeof(GPU()) ?  CuArray{Bool}(undef,(grid.nx, grid.ny, grid.nz)) : BitArray(undef,(grid.nx, grid.ny, grid.nz));
    
  VP_Vars( ux,  uy,  uz,  bx,  by,  bz,
          uxh, uyh, uzh, bxh, byh, bzh,
          nonlin1, nonlinh1, Fx, Fy, Fgx, Fgy,
          A,  Φ1, Φ2, Ω);
end

function Setupfos1s2(grid::AbstractGrid;kf = 1)
  # The 22 conponment 
  fox,foy,foz = zeros(Int32,22),zeros(Int32,22),zeros(Int32,22);
#=  k = 1;
  for θ ∈ [-5,0,5].*π/180 #anisotropic turbulence injection
    for ϕ ∈ [-25,-15,-5,0,5,15,25].*π/180
      fox[k] = round(Int32,kf*cos(θ));
      foy[k] = round(Int32,kf*sin(θ)*sin(ϕ));
      foz[k] = round(Int32,kf*sin(θ)*cos(ϕ));
      k+=1;
    end
  end
  fox[22] = 0;
  foy[22] = kf;
  foz[22] = 0;
=#
  #for k = 1:22
  #  println("fox[$k] = "*string(fox[k])*",foy[$k] = "*string(foy[k])*",foz[$k] = "*string(foz[k]));
  #end
  
  fox[1]=  2; foy[1]=  1; foz[1]=  1;
  fox[2]=  2; foy[2]=  1; foz[2]= -1;
  fox[3]=  2; foy[3]= -1; foz[3]=  1;
  fox[4]=  2; foy[4]= -1; foz[4]= -1;
  fox[5]=  1; foy[5]=  2; foz[5]=  1;
  fox[6]=  1; foy[6]=  2; foz[6]= -1;
  fox[7]=  1; foy[7]=  1; foz[7]=  2;
  fox[8]=  1; foy[8]=  1; foz[8]= -2;
  fox[9]=  1; foy[9]= -1; foz[9]=  2;
  fox[1]=  1; foy[10]=-1; foz[10]=-2;
  fox[11]= 1; foy[11]=-2; foz[11]= 1;
  fox[12]= 1; foy[12]=-2; foz[12]=-1;
  fox[13]= 2; foy[13]= 0; foz[13]= 0;
  fox[14]= 3; foy[14]= 0; foz[14]= 0;
  fox[15]= 0; foy[15]= 2; foz[15]= 0;
  fox[16]= 0; foy[16]= 3; foz[16]= 0;
  fox[17]= 0; foy[17]= 0; foz[17]= 2;
  fox[18]= 0; foy[18]= 0; foz[18]= 3;
  fox[19]= 2; foy[19]= 2; foz[19]= 2;
  fox[20]= 2; foy[20]= 2; foz[20]=-2;
  fox[21]= 2; foy[21]=-2; foz[21]= 2;
  fox[22]= 2; foy[22]=-2; foz[22]=-2;
  

  fo = zeros(Int32,3,22);
  for k = 1:22
    fo[1,k] = round(Int32,(kf/3*fox[k]));
    fo[2,k] = round(Int32,(kf/3*foy[k]));
    fo[3,k] = round(Int32,(kf/3*foz[k]));
  end
  # Set up vector set s1 s2 that ⊥ k_f
  s1,s2 = zeros(3,22),zeros(3,22);
  k_component = 22;
    for k_i = 1:k_component
      # index 1,2,3 -> i,j,k direction
      rkx,rky,rkz = fox[k_i],foy[k_i],foz[k_i];
      ryz = √( rky^2 +rkz^2 );
      rxyz= √( rkx^2 +rky^2 +rkz^2);
      if (ryz == 0.0)
           s1[1,k_i]=0.0
           s1[2,k_i]=1.0
           s1[3,k_i]=0.0
           s2[1,k_i]=0.0
           s2[2,k_i]=0.0
           s2[3,k_i]=1.0
       else
       s1[1,k_i] =  0.0
       s1[2,k_i] =  rkz / ryz
       s1[3,k_i] = -rky / ryz

       s2[1,k_i] = -ryz / rxyz
       s2[2,k_i] =  rkx*rky / rxyz / ryz
       s2[3,k_i] =  rkx*rkz / rxyz / ryz
    end
  end

  return fo,s1,s2;
end

function setupChovars!(vars;A0=1.0)
    rand1(n::Int) = 2 .*(rand(n).-0.5);
    Φ1,Φ2,A = vars.Φ1,vars.Φ2,vars.A;
    k_conponment = length(Φ1);
    
    copyto!(Φ1,π*rand1(k_conponment));
    copyto!(Φ2,π*rand1(k_conponment));
    #just 1 conponment for now
    copyto!(A, [A0]);
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