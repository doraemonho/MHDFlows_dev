module MHDSolver
# ----------
# Navier–Stokes Solver for 3D Magnetohydrodynamics Problem
# ----------

export 
	UᵢUpdate!,
	BᵢUpdate!,
	MHDcalcN_advection!,
	MHDupdatevars!

using
  CUDA,
  TimerOutputs

using LinearAlgebra: mul!, ldiv!
include("VPSolver.jl")

# δ notation
δ(a::Int,b::Int) = ( a == b ? 1 : 0 )
# ϵ notation
ϵ(i::Int,j::Int,k::Int) = (i - j)*(j - k)*(k - i)/2

# checking function of VP method
VP_is_turned_on(params) = hasproperty(params,:U₀x);

function UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="x")

  if direction == "x"

  	# a = {1,2,3} -> {x,y,z} direction
  	a    = 1
  	kₐ   = grid.kr
  	k⁻²  = grid.invKrsq
  	∂uᵢh∂t = @view N[:,:,:,params.ux_ind]

  elseif direction == "y"

  	a    = 2
  	kₐ   = grid.l
  	k⁻²  = grid.invKrsq
  	∂uᵢh∂t = @view N[:,:,:,params.uy_ind]

  elseif direction == "z"
  	a    = 3
  	kₐ   = grid.m
  	k⁻²  = grid.invKrsq
  	∂uᵢh∂t = @view N[:,:,:,params.uz_ind]

  else

  	error("Warning : Unknown direction is declerad")

  end
  #idea : we are computing ∂uᵢh∂t = im*kᵢ*(δₐⱼ - kₐkⱼk⁻²)*(bᵢbⱼ - uᵢuⱼh) 
  #  as uᵢuⱼ = uⱼuᵢ in our case
  #     1  2  3
  #   1 11 12 13
  #   2 21 22 23 , part of computation is repeated, 11(1),12(2),13(2),22(1),23(2),33(1)
  #   3 31 32 33
  #   Their only difference for u_ij is the advection part
  @. ∂uᵢh∂t*= 0;
  for (bᵢ,uᵢ,kᵢ,i) ∈ zip((vars.bx,vars.by,vars.bz),(vars.ux,vars.uy,vars.uz),(grid.kr,grid.l,grid.m),(1,2,3))
    for (bⱼ,uⱼ,kⱼ,j) ∈ zip((vars.bx,vars.by,vars.bz),(vars.ux,vars.uy,vars.uz),(grid.kr,grid.l,grid.m),(1, 2, 3))
      if j >= i
        # Initialization
        @. vars.nonlin1  *= 0
        @. vars.nonlinh1 *= 0
        bᵢbⱼ_minus_uᵢuⱼ  = vars.nonlin1  
        bᵢbⱼ_minus_uᵢuⱼh = vars.nonlinh1

        # Perform Computation in Real space
        @. bᵢbⱼ_minus_uᵢuⱼ = bᵢ*bⱼ - uᵢ*uⱼ
        mul!(bᵢbⱼ_minus_uᵢuⱼh, grid.rfftplan, bᵢbⱼ_minus_uᵢuⱼ)

        # Perform the Actual Advection update
        @. ∂uᵢh∂t += im*kᵢ*(δ(a,j)-kₐ*kⱼ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh
        if i != j  # repeat the calculation for u_ij
          @. ∂uᵢh∂t += im*kⱼ*(δ(a,i)-kₐ*kᵢ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh
        end
      end
    end
  end

  # Updating the solid domain if VP flag is ON
  if VP_is_turned_on(params) 
    VPSolver.VP_UᵢUpdate!(∂uᵢh∂t, kₐ.*k⁻², a, clock, vars, params, grid)
  end

  #Compute the diffusion term  - νk^2 u_i
  uᵢ = direction == "x" ? vars.ux : direction == "y" ? vars.uy : vars.uz;
  uᵢh = vars.nonlinh1
  mul!(uᵢh, grid.rfftplan, uᵢ)
  @. ∂uᵢh∂t += -grid.Krsq*params.ν*uᵢh

  return nothing
    
end

# B function
function BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")

	#To Update B_i, we have two terms to compute:
	# ∂B_i/∂t = im ∑_j k_j*(b_iu_j - u_ib_j)  - η k^2 B_i
	#We split it into two part for sparating the computation.

  # declare the var u_i, b_i for computation
	if direction == "x"
    a   = 1;
    kₐ  = grid.kr;
    k⁻² = grid.invKrsq;
		uᵢ  = vars.ux;
		bᵢ  = vars.bx; 
		∂Bᵢh∂t = @view N[:,:,:,params.bx_ind];

	elseif direction == "y"
    a   = 2;
    kₐ  = grid.l;
    k⁻² = grid.invKrsq;
		uᵢ  = vars.uy;
		bᵢ  = vars.by; 
		∂Bᵢh∂t = @view N[:,:,:,params.by_ind];

	elseif direction == "z"
    a   = 3;
    kₐ  = grid.m;
    k⁻² = grid.invKrsq;
		uᵢ  = vars.uz;
		bᵢ  = vars.bz; 
		∂Bᵢh∂t = @view N[:,:,:,params.bz_ind];

	else

		@warn "Warning : Unknown direction is declerad"

	end

  @. ∂Bᵢh∂t*= 0
  uᵢbⱼ_minus_bᵢuⱼ  = vars.nonlin1        
  uᵢbⱼ_minus_bᵢuⱼh = vars.nonlinh1
  #Compute the first term, im ∑_j k_j*(b_iu_j - u_ib_j)
  for (bⱼ,uⱼ,kⱼ,j) ∈ zip((vars.bx,vars.by,vars.bz),(vars.ux,vars.uy,vars.uz),(grid.kr,grid.l,grid.m),(1,2,3))
    if a != j
      # Perform Computation in Real space
      @. uᵢbⱼ_minus_bᵢuⱼ = uᵢ*bⱼ - bᵢ*uⱼ
      
      mul!(uᵢbⱼ_minus_bᵢuⱼh, grid.rfftplan, uᵢbⱼ_minus_bᵢuⱼ);

      # Perform the Actual Advection update
      @. ∂Bᵢh∂t += im*kⱼ*uᵢbⱼ_minus_bᵢuⱼh  

    end
  end

  # Updating the solid domain if VP flag is ON
  if VP_is_turned_on(params) 
    VPSolver.VP_BᵢUpdate!(∂Bᵢh∂t, kₐ.*k⁻², a, clock, vars, params, grid)
  end

  #Compute the diffusion term  - ηk^2 B_i
  bᵢh = vars.nonlinh1;
  mul!(bᵢh, grid.rfftplan, bᵢ); 
  @. ∂Bᵢh∂t += -grid.Krsq*params.η*bᵢh
    
  return nothing

end

# # Update ideal magnetic helicity equation with coulomb guage version ( ∇⋅ A = 0 )
function HUpdate!(N, sol, t, clock, vars, params, grid)
  #To Update ideal H, we have to compute:
  # ∂H/∂t = -∇⋅(Jₕ) 
  # where J_H = A × (v × B - ηJ + ∇Φ) and if we let E = v × B - ηJ + ∇Φ
  # using vector idientities, we arrived at
  # ∇⋅(A × E) = (∇ × A) ⋅ E - (∇ × E) ⋅ A = B ⋅ E - (∇ × E) ⋅ A  
  # we have two terms to compute
  # ∂H/∂t = - (B ⋅ E - (∇ × E) ⋅ A ) =  -B ⋅ E + (∇ × E) ⋅ A 

  # define all the vars & sketch vars
  ∂H∂t = @view N[:,:,:,7]
  
  # Imag space vars
  # ∇XE already computed in the induction equation 
  @views ∇XExh, ∇XEyh, ∇XEzh =  N[:,:,:,params.bx_ind],   N[:,:,:,params.by_ind],   N[:,:,:,params.bz_ind]
  @views    bxh,   byh,   bzh = sol[:,:,:,params.bx_ind], sol[:,:,:,params.by_ind], sol[:,:,:,params.bz_ind]
  # Real space vars
  bx, by, bz   = vars.bx, vars.by, vars.bz
  vx, vy, vz   = vars.ux, vars.uy, vars.uz
  @views Ax, Ay, Az   = params.usr_params.sk[:,:,:,1], params.usr_params.sk[:,:,:,2], params.usr_params.sk[:,:,:,3]
  @views Jx, Jy, Jz   = params.usr_params.sk[:,:,:,4], params.usr_params.sk[:,:,:,5], params.usr_params.sk[:,:,:,6]
  @views Ex, Ey, Ez   = params.usr_params.Dᵢⱼ[:,:,:,1,1], params.usr_params.Dᵢⱼ[:,:,:,1,2], params.usr_params.Dᵢⱼ[:,:,:,1,3]

  η   = params.η
  k⁻² = grid.invKrsq
  kx, ky, kz = grid.kr, grid.l, grid.m

  #compute J & A 
  Jxh = Jyh = Jzh = vars.nonlinh1 
  #--------------- X dir J & A ---------------------#
  @. Jxh = im*(ky*bzh - kz*byh)
  ldiv!(Jx, grid.rfftplan, deepcopy(Jxh))
  @. Jxh *= k⁻² # using vector idientites with coulomb guage, A = k²⋅J 
  ldiv!(Ax, grid.rfftplan, deepcopy(Jxh))
  #--------------- Y dir J & A ---------------------#
  @. Jyh = im*(kz*bxh - kx*bzh)
  ldiv!(Jy, grid.rfftplan, deepcopy(Jyh))
  @. Jyh *= k⁻² # using vector idientites with coulomb guage, A = k²⋅J 
  ldiv!(Ay, grid.rfftplan, deepcopy(Jyh))
  #--------------- Z dir J & A ---------------------#
  @. Jzh = im*(kx*byh - ky*bxh)
  ldiv!(Jz, grid.rfftplan, deepcopy(Jzh))
  @. Jzh *= k⁻² # using vector idientites with coulomb guage, A = k²⋅J 
  ldiv!(Az, grid.rfftplan, deepcopy(Jzh))

  #compute E
  @. Ex = vy*bz - vz*by - η*Jx
  @. Ey = vz*bx - vx*bz - η*Jy
  @. Ez = vx*by - vy*bx - η*Jz

  # work out the actual advection
  @. ∂H∂t = 0
  ∇XE_dot_A = B_dot_E = vars.nonlin1
  ∇XE_dot_Ah = B_dot_Eh = vars.nonlinh1
  for (∇XEᵢh, Aᵢ, Bᵢ, Eᵢ) ∈ zip((∇XExh,∇XEyh,∇XEzh), (Ax, Ay, Az), (bx,by,bz), (Ex,Ey,Ez))
    # compute (∇ × E) ⋅ A 
    ldiv!(∇XE_dot_A, grid.rfftplan, deepcopy(∇XEᵢh))
    @. ∇XE_dot_A *= Aᵢ
    mul!(∇XE_dot_Ah, grid.rfftplan,∇XE_dot_A)
    @. ∂H∂t += ∇XE_dot_Ah

    # compute B ⋅ E
    @. B_dot_E = Bᵢ*Eᵢ
    mul!(B_dot_Eh, grid.rfftplan, B_dot_E)    
    @. ∂H∂t -= B_dot_Eh
  end

  return nothing 

end


# B function for EMHD system
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

  #Update B Real Conponment
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind]))

  return nothing
end

function MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]))
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]))
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]))
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 

  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")

  #Update B Advection
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z") 

  return nothing
end

function HMHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]))
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]))
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]))
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 

  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")

  #Update B Advection
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z") 

  #Update H Advection
  HUpdate!(N, sol, t, clock, vars, params, grid)  

  return nothing
end

end
