module EMHDSolver
# ----------
# Implicit Solver for 3D Electron Magnetohydrodynamics Problem
# ----------

using LinearAlgebra: mul!, ldiv!

# Advection function for EMHD system
# For E-MHD system, the induction will be changed into
#  ∂B/∂t = -dᵢ * ∇× [ (∇× B) × B ] + η ∇²B
# In this function, we will implement the equation and assume dᵢ = 1
function EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

  # To Update B_i, we have to first break down the equation :
  # ∂B/∂t  = - ∇× [ (∇× B) × B ] + η ∇²B
  # Let A = (∇× B). By using vector calculus identities, we have
  # ∂B/∂t  = - [ (∇ ⋅ B + B ⋅ ∇)A - (∇ ⋅ A + A ⋅ ∇)B ]  + η ∇²B
  # Using ∇ ⋅ B  = 0 and vector calculus identities ∇⋅(∇× B) = 0, we finally get the expression
  # ∂B/∂t  = - [(B ⋅ ∇)A - (A ⋅ ∇)B ]  + η ∇²B =  (A ⋅ ∇)B - (B ⋅ ∇)A  + η ∇²B
  # For any direction i, we will have the following expression in k-space
  # 𝔉(∂Bᵢ/∂t)  = 𝔉[(Aⱼ∂ⱼ)Bᵢ - Bⱼ∂ⱼAᵢ] -  k²𝔉(B)
  # To compute the first term in RHS, we break it into three step
  # 1. compute real space term ∂ⱼBᵢ using spectral method
  # 2. compute Aⱼ∂ⱼBᵢ using pseudo spectral method
  # 3. add the answer to 𝔉(∂Bᵢ/∂t) 
  #

  # declare the var u_i, b_i for computation
  if direction == "x"
    a   = 1
    kₐ  = grid.kr
    Aᵢ  = vars.∇XBᵢ
    Aᵢh = vars.∇XBᵢh
    bᵢ  = vars.bx 
    bᵢh = @view sol[:,:,:,params.bx_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.bx_ind]

  elseif direction == "y"
    a   = 2
    kₐ  = grid.l
    Aᵢ  = vars.∇XBⱼ
    Aᵢh = vars.∇XBⱼh
    bᵢ  = vars.by 
    bᵢh = @view sol[:,:,:,params.by_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.by_ind]

  elseif direction == "z"
    a   = 3
    kₐ  = grid.m
    Aᵢ  = vars.∇XBₖ
    Aᵢh = vars.∇XBₖh
    bᵢ  = vars.bz 
    bᵢh = @view sol[:,:,:,params.bz_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.bz_ind]
  else

    @warn "Warning : Unknown direction is declerad"

  end

  A₁  = vars.∇XBᵢ
  A₂  = vars.∇XBⱼ
  A₃  = vars.∇XBₖ

  # define the sketch array
  Bᵢh   = vars.nonlinh1
  ∂ⱼAᵢ  = ∂ⱼBᵢ  = vars.nonlin1
  Bⱼ∂ⱼAᵢ= Aⱼ∂ⱼBᵢ= vars.nonlin1
  ∂ⱼAᵢh = ∂ⱼBᵢh = vars.nonlinh1
  Bⱼ∂ⱼAᵢh = Aⱼ∂ⱼBᵢh = vars.nonlinh1

  @. ∂Bᵢh∂t*= 0
  for (bⱼ,Aⱼ,kⱼ) ∈ zip((vars.bx,vars.by,vars.bz),(A₁,A₂,A₃),(grid.kr,grid.l,grid.m))
    
    # first step
    @. ∂ⱼAᵢh = im*kⱼ*Aᵢh
    ldiv!(∂ⱼAᵢ, grid.rfftplan, deepcopy(∂ⱼAᵢh))
    # second step
    @. Bⱼ∂ⱼAᵢ = bⱼ*∂ⱼAᵢ
    @. Bⱼ∂ⱼAᵢh = 0
    mul!(Bⱼ∂ⱼAᵢh, grid.rfftplan, Bⱼ∂ⱼAᵢ)
    # final step
    @. ∂Bᵢh∂t -= Bⱼ∂ⱼAᵢh

    # first step
    @. ∂ⱼBᵢ = 0
    @. ∂ⱼBᵢh = im*kⱼ*bᵢh
    ldiv!(∂ⱼBᵢ, grid.rfftplan, deepcopy(∂ⱼBᵢh))
    # second step
    @. Aⱼ∂ⱼBᵢ = Aⱼ*∂ⱼBᵢ
    @. Aⱼ∂ⱼBᵢh = 0
    mul!(Aⱼ∂ⱼBᵢh, grid.rfftplan, Aⱼ∂ⱼBᵢ)
    # final step
    @. ∂Bᵢh∂t += Aⱼ∂ⱼBᵢh
    
  end

  return nothing
  
end


# Compute the ∇XB term
function Get∇XB!(sol, vars, params, grid)

  # ∇XB = im*( k × B )ₖ = im*ϵ_ijk kᵢ Bⱼ

  # define the variables
  k₁,k₂,k₃ = grid.kr,grid.l,grid.m;
  B₁h = @view sol[:,:,:,params.bx_ind]
  B₂h = @view sol[:,:,:,params.by_ind]
  B₃h = @view sol[:,:,:,params.bz_ind]
  A₁  = vars.∇XBᵢ
  A₂  = vars.∇XBⱼ
  A₃  = vars.∇XBₖ

  # Way 2 of appling Curl
  @. vars.∇XBᵢh = im*(k₂*B₃h - k₃*B₂h)
  ldiv!(A₁, grid.rfftplan, deepcopy(vars.∇XBᵢh))  

  @. vars.∇XBⱼh = im*(k₃*B₁h - k₁*B₃h)
  ldiv!(A₂, grid.rfftplan, deepcopy(vars.∇XBⱼh))  

  @. vars.∇XBₖh = im*(k₁*B₂h - k₂*B₁h)
  ldiv!(A₃, grid.rfftplan, deepcopy(vars.∇XBₖh))  

  return nothing
end

function EMHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update B Advection
  Get∇XB!(sol, vars, params, grid)
  EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  EMHD_BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")

  #Update diffusion
  @. N -= params.η*grid.Krsq*sol

  #Update B Real Conponment
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind]))

  return nothing
end

end