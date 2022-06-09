module pgenAlfven
#Problem Gernerator for setting Up the problem 

using 
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

include("MHDSolver.jl")
include("datastructure.jl")

export AlfProblem

MHDcalcN_advection!  = MHDSolver.MHDcalcN_advection!;

nothingfunction(args...) = nothing;

function AlfProblem(dev::Device=CPU();
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
             calcF = nothingfunction,
                A0 = 1.0,
                kf = 1.0,
  # Float type and dealiasing
                 T = Float32)

  grid = ThreeDGrid(dev, nx, Lx, ny, Ly, nz, Lz; T=T)

  # set up the function for this three
  fo,s1,s2 = Setupfos1s2(grid;kf=kf);
  params = ChoParams{T}(ν, η, nν, 1, 2, 3, 4, 5, 6, calcF, fo, s1,s2)

  vars = SetChoVars(dev, grid);
  setupChovars!(vars;A0=A0);

  equation = Equation_with_forcing(dev, params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)

end

abstract type MHDVars <: AbstractVars end
struct ChoVars{Aphys, Atrans, Avars1, Avars2} <: MHDVars
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
    "Forcing Amplitude"
     A  :: Avars1
    "Random Phase 1"
     Φ1 :: Avars2
    "Random Phase 2"
     Φ2 :: Avars2
end
struct ChoParams{T} <: AbstractParams
    
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

end

function SetChoVars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1
  @devzeros Dev T (22) Φ1 Φ2
  @devzeros Dev T (1) A
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) uxh uyh uzh bxh byh bzh nonlinh1
  
  ChoVars( ux,  uy,  uz,  bx,  by,  bz,
          uxh, uyh, uzh, bxh, byh, bzh,
          nonlin1, nonlinh1,
          A,  Φ1, Φ2);
end

function Setupfos1s2(grid::AbstractGrid;kf = 15)
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
  fo[1,:] .= fox;
  fo[2,:] .= foy;
  fo[3,:] .= foz;

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

function Equation_with_forcing(dev,params::ChoParams, grid::AbstractGrid)
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr,grid.nm,grid.nk, 6));
    
  return FourierFlows.Equation(L,MHDcalcN!, grid)
end


end