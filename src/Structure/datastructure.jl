# ----------
# Module for Setting up the data structure for HD and MHD problem
# ----------

function SetMHDVars(::Dev, grid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm)  nonlinh1
  
  return MVars( ux,  uy,  uz,  bx,  by,  bz,
                nonlin1, nonlinh1, usr_vars);
end

function SetHMHDVars(::Dev, grid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz  bx  by bz nonlin1
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) nonlinh1 jxh jyh jzh Φh
  
  return HMVars(  ux,  uy,  uz, bx,  by,  bz, 
                 jxh, jyh, jzh, Φh,
                 nonlin1, nonlinh1, usr_vars);
end


function SetEMHDVars(::Dev, grid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) bx  by bz ∇XBx ∇XBy ∇XBz ∇XBXB₁ ∇XBXB₂ ∇XBXB₃ nonlin1
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm)  nonlinh1 ∇XBxh ∇XByh ∇XBzh ∇XBXB₁h ∇XBXB₂h ∇XBXB₃h
  
  return EMVars(bx,  by,  bz,
                ∇XBx, ∇XBy, ∇XBz, ∇XBXB₁, ∇XBXB₂, ∇XBXB₃,
                ∇XBxh, ∇XByh, ∇XBzh, ∇XBXB₁h, ∇XBXB₂h, ∇XBXB₃h,
                nonlin1, nonlinh1, usr_vars);
end


function SetHDVars(::Dev, grid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ux  uy  uz nonlin1 
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) nonlinh1
  
  return HVars( ux,  uy,  uz, 
                nonlin1, nonlinh1, usr_vars);
end

function SetCMHDVars(::Dev, grid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ρ ux  uy  uz  bx  by bz nonlin1 nonlin2
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm)  uxh uyh uzh nonlinh1 nonlinh2
  
  return CMVars( ρ, ux,  uy,  uz,  bx,  by,  bz, uxh, uyh, uzh,
                nonlin1, nonlinh1, nonlin2, nonlinh2, usr_vars);
end

function SetCHDVars(::Dev, grid, usr_vars) where Dev
  T = eltype(grid)
    
  @devzeros Dev T (grid.nx, grid.ny, grid.nz) ρ ux uy  uz nonlin1 nonlin2
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, grid.nm) uxh uyh uzh nonlinh1 nonlinh2
  
  return CHVars( ρ, ux,  uy,  uz, uxh, uyh, uzh, 
                 nonlin1, nonlinh1, nonlin2, nonlinh2, usr_vars);
end

# Functions of setting up the Vars and Params struct
function SetVars(dev, grid, usr_vars; B = false, H = false, E = false, VP = false, C =false)
  if C 
    setvars = ifelse(B,SetCMHDVars,SetCHDVars)
  elseif E
    setvars = SetEMHDVars
  elseif H
    setvars = SetHMHDVars    
  else
    setvars = ifelse(B,SetMHDVars,SetHDVars)
  end
  return setvars(dev, grid, usr_vars)
end

 function SetParams(::Dev, grid, calcF::Function, usr_params;
                     H = false, B = false, VP = false, C = false, S = false, E = false,
                     cₛ = 0, ν = 0, η = 0, nν = 0, nη = 0, hν = 0, hη = 0) where Dev
  T = eltype(grid);
  usr_param = typeof(usr_params)
  
  # define the debug timer
  to = TimerOutput();

  if (B) || (H) || (E)
    if (VP)  
      @devzeros Dev T (grid.nx, grid.ny, grid.nz) χ U₀x U₀y U₀z B₀x B₀y B₀z
      params = MHDParams_VP(ν, η, nν, nη, hν, hη, 1, 2, 3, 4, 5, 6, calcF, χ, U₀x, U₀y, U₀z, B₀x, B₀y, B₀z, usr_params, to)
    elseif (C)
      params = CMHDParams(cₛ,ν, η, nν, nη, 1, 2, 3, 4, 5, 6, 7, calcF, usr_params, to);
    elseif (S)
      shear_params = GetShearParams(Dev, grid, B; ν=ν, η=η);
      params = MHDParams(0.0, 0.0, nν, nη, 1, 2, 3, 4, 5, 6, calcF, shear_params, to);
    elseif (E)
      params = EMHDParams(η, nη, 1, 2, 3, calcF, usr_params, to);
    else   
      params = MHDParams(ν, η, nν, nη, hν, hη, 1, 2, 3, 4, 5, 6, calcF, usr_params, to);
    end
  else
    if (VP)
      @devzeros Dev T (grid.nx, grid.ny, grid.nz) χ U₀x U₀y U₀z
      params = HDParams_VP(ν, nν, hη,  1, 2, 3, calcF, χ, U₀x, U₀y, U₀z, usr_params, to);
    elseif (C)
      params = CHDParams(cₛ, ν, nν, 1, 2, 3, 4, calcF, usr_params, to);
    elseif (S)
      shear_params = GetShearParams(Dev, grid, B; ν=ν, η=0.0);
      params = HDParams(0.0, nν, 1, 2, 3, calcF, shear_params, to);
    else
      params = HDParams(ν, nν, hη, 1, 2, 3, calcF, usr_params, to);
    end
  end

  return params

end

function GetShearParams(dev, grid, B; ν = 0.0, η = 0.0)
  T = eltype(grid)

  @devzeros dev T (grid.nx, grid.ny, grid.nz) U₀x U₀y
  @devzeros dev Complex{T} (grid.nkr, grid.nl, grid.nm)  U₀xh U₀yh
  ky₀ = copy(grid.l);
  iky = copy(grid.l1D)
  k2xz= @. grid.kr^2 + grid.m^2
  Nₗ = ifelse(B,6,3)
  @devzeros dev Complex{T} (grid.nkr, grid.nl, grid.nm, Nₗ)  tmp

  return SParams(T(0.0), T(0.0), T(0.0), T(ν), ky₀, k2xz, iky, U₀x, U₀y, U₀xh, U₀yh, tmp)
end

mutable struct SParams{A1Daxis,A2Daxis, Aphys, Atrans, Atmp} <: AbstractParams
  "shear rate = dlnΩ/dlnr"
    q  :: AbstractFloat
  "built-in time"
    τ  :: AbstractFloat
  "Remapping peroid"
    τΩ :: AbstractFloat
  "diffusion Coef."
     ν :: AbstractFloat
  "spectral ky at t = 0"
   ky₀ :: A2Daxis
  "spectral kx² + kz² at t = 0"
   k2xz:: A2Daxis
  "spectral ky in 1D at t = 0"
   iky :: A1Daxis

  "Background shear velocity in real/spectral space"
  U₀x   :: Aphys
  U₀y   :: Aphys
  U₀xh  :: Atrans
  U₀yh  :: Atrans

  "Sketch array"
   tmp  :: Atmp
end