# ----------
# Module for Setting up the data structure for HD and MHD problem
# ----------

function SetMHDVars(::Dev, grid::AbstractGrid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm)  nonlinh1
  
  return MVars( ux,  uy,  uz,  bx,  by,  bz,
                nonlin1, nonlinh1, usr_vars);
end

function SetHDVars(::Dev, grid::AbstractGrid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz nonlin1 
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) nonlinh1
  
  return HVars( ux,  uy,  uz, 
                nonlin1, nonlinh1, usr_vars);
end

function SetCMHDVars(::Dev, grid::AbstractGrid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ρ ux  uy  uz  bx  by bz nonlin1 nonlin2
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm)  uxh uyh uzh nonlinh1 nonlinh2
  
  return CMVars( ρ, ux,  uy,  uz,  bx,  by,  bz, uxh, uyh, uzh,
                nonlin1, nonlinh1, nonlin2, nonlinh2, usr_vars);
end

function SetCHDVars(::Dev, grid::AbstractGrid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ρ ux uy  uz nonlin1 nonlin2
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) uxh uyh uzh nonlinh1 nonlinh2
  
  return CHVars( ρ, ux,  uy,  uz, uxh, uyh, uzh, 
                 nonlin1, nonlinh1, nonlin2, nonlinh2, usr_vars);
end

# Functions of setting up the Vars and Params struct
function SetVars(dev, grid, usr_vars; B = false, VP = false, C =false)
  if C 
    setvars = ifelse(B,SetCMHDVars,SetCHDVars);
  else
    setvars = ifelse(B,SetMHDVars,SetHDVars);
  end
  return setvars(dev, grid, usr_vars);
end

 function SetParams(::Dev, grid::AbstractGrid, calcF::Function, usr_params;
                     B = false, VP = false, C = false, S = false, cₛ = 0, ν = 0, η = 0, nν = 0, nη = 0) where Dev
  T = eltype(grid);
  usr_param = typeof(usr_params)
  
  # define the debug timer
  to = TimerOutput();

  if (B)
    if (VP)
      @devzeros Dev T (grid.nx, grid.ny, grid.nz) χ U₀x U₀y U₀z B₀x B₀y B₀z
      params = MHDParams_VP(ν, η, nν, nη, 1, 2, 3, 4, 5, 6, calcF, χ, U₀x, U₀y, U₀z, B₀x, B₀y, B₀z, usr_params, to)
    elseif (C)
      params = CMHDParams(cₛ,ν, η, nν, nη, 1, 2, 3, 4, 5, 6, 7, calcF, usr_params, to);
    elseif (S)
      shear_params = GetShearParams(Dev, grid, B);
      params = MHDParams(ν, η, nν, nη, 1, 2, 3, 4, 5, 6, calcF, shear_params, to);
    else
      params = MHDParams(ν, η, nν, nη, 1, 2, 3, 4, 5, 6, calcF, usr_params, to);
    end
  else
    if (VP)
      @devzeros Dev T (grid.nx, grid.ny, grid.nz) χ U₀x U₀y U₀z
      params = HDParams_VP(ν, nν, 1, 2, 3, calcF, χ, U₀x, U₀y, U₀z, usr_params, to);
    elseif (C)
      params = CHDParams(cₛ, ν, nν, 1, 2, 3, 4, calcF, usr_params, to);
    elseif (S)
      shear_params = GetShearParams(Dev, grid, B);
      params = MHDParams(ν, η, nν, nη, 1, 2, 3, 4, 5, 6, calcF, shear_params, to);
    else
      params = HDParams(ν, nν, 1, 2, 3, calcF, usr_params, to);
    end
  end

  return params

end

function GetShearParams(dev, grid, B)
  T = eltype(grid)

  @devzeros dev T (grid.nx, grid.ny, grid.nz) U₀x U₀y
  @devzeros dev Complex{T} (grid.nkr, grid.nl, grid.nm)  U₀xh U₀yh
  ky₀ = copy(grid.l);

  Nₗ = ifelse(B,6,3)
  @devzeros dev Complex{T} (grid.nkr, grid.nl, grid.nm, Nₗ)  tmp

  return SParams(0.0, 0.0, 0.0, 0.0, ky₀, U₀x, U₀y, U₀xh, U₀yh, tmp)
end

mutable struct SParams{ATs, Aphys, Atrans, Atmp} <: AbstractParams
  "shear rate = dlnΩ/dlnr"
    q  :: AbstractFloat
  "Anagular velocty"
    Ω  :: AbstractFloat
  "built-in time"
    τ  :: AbstractFloat
  "Remapping peroid"
    τΩ :: AbstractFloat
  "spectral ky at t = 0"
  ky₀ :: ATs
  
  "Background shear velocity in real/spectral space"
  U₀x   :: Aphys
  U₀y   :: Aphys
  U₀xh  :: Atrans
  U₀yh  :: Atrans

  "Sketch array"
   tmp  :: Atmp
end