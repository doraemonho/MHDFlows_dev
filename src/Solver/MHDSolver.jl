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

# Vector Potential function instead of magnetic field (coulomb guage version)
function AᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
  #To Update A_i, we have two terms to compute:
  # ∂A_i/∂t = v × B - η ∇ × B - ∇Φ


 # declare the var u_i, b_i for computation
  if direction == "x"
    i   = 1
    kᵢ  = grid.kr
    Jᵢh    = vars.jxh
    ∂Aᵢh∂t = @view N[:,:,:,params.bx_ind];

  elseif direction == "y"
    i   = 2;
    kᵢ  = grid.l;
    Jᵢh    = vars.jyh
    ∂Aᵢh∂t = @view N[:,:,:,params.by_ind];

  elseif direction == "z"
    i   = 3;
    kᵢ  = grid.m;
    Jᵢh    = vars.jzh
    ∂Aᵢh∂t = @view N[:,:,:,params.bz_ind];
  else
    @warn "Warning : Unknown direction is declerad"
  end

  @. ∂Aᵢh∂t .= 0
  ε_ijkuⱼbₖ  = vars.nonlin1        
  ε_ijkuⱼbₖh = vars.nonlinh1

  # compute the term v × B =  ε_ijk v_j b_k
  for (uⱼ,j) ∈ zip((vars.ux,vars.uy,vars.uz),(1,2,3))
    for (bₖ,k) ∈ zip((vars.bx,vars.by,vars.bz),(1,2,3))
      if ϵ(i,j,k) > 0.0
        ϵ_ijk = ϵ(i,j,k)
        @. ε_ijkuⱼbₖ = ϵ_ijk*uⱼ*bₖ
        mul!(ε_ijkuⱼbₖh, grid.rfftplan, ε_ijkuⱼbₖ)
        @. ∂Aᵢh∂t += ε_ijkuⱼbₖh
      end
    end
  end

  # compute η ∇ × B
  @. ∂Aᵢh∂t -= params.η*Jᵢh

  # compute the ∇Φ term using ∇²Φ = ∇⋅(v×B - ηJ) 
  @. ∂Aᵢh∂t -= kᵢ*vars.Φh

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
    bᵢ  = vars.bx 
    bᵢh = @view sol[:,:,:,params.bx_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.bx_ind]

  elseif direction == "y"
    a   = 2
    kₐ  = grid.l
    Aᵢ  = vars.∇XBⱼ
    bᵢ  = vars.by 
    bᵢh = @view sol[:,:,:,params.by_ind]
    ∂Bᵢh∂t = @view N[:,:,:,params.by_ind]

  elseif direction == "z"
    a   = 3
    kₐ  = grid.m
    Aᵢ  = vars.∇XBₖ
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
  ∂ⱼAᵢ  = ∂ⱼBᵢ  = vars.nonlin1
  Bⱼ∂ⱼAᵢ= Aⱼ∂ⱼBᵢ= vars.nonlin1
  Aᵢh   = Bᵢh   = vars.nonlinh1
  ∂ⱼAᵢh = ∂ⱼBᵢh = vars.nonlinh1
  Bⱼ∂ⱼAᵢh = Aⱼ∂ⱼBᵢh = vars.nonlinh1

  @. ∂Bᵢh∂t*= 0;
  for (bⱼ,Aⱼ,kⱼ) ∈ zip((vars.bx,vars.by,vars.bz),(A₁,A₂,A₃),(grid.kr,grid.l,grid.m))
    
    # first step
    @. Aᵢh = 0
    mul!(Aᵢh, grid.rfftplan, Aᵢ)
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
  CBᵢh = vars.nonlinh1
  @. CBᵢh = im*(k₂*B₃h - k₃*B₂h)
  ldiv!(A₁, grid.rfftplan, CBᵢh)  

  @. CBᵢh = im*(k₃*B₁h - k₁*B₃h)
  ldiv!(A₂, grid.rfftplan, CBᵢh)  

  @. CBᵢh = im*(k₁*B₂h - k₂*B₁h)
  ldiv!(A₃, grid.rfftplan, CBᵢh)  

  return nothing
end

# Compute the B from ∇XA term
function UpdateB!(sol, vars, params, grid)

  # ∇XB = im*( k × B )ₖ = im*ϵ_ijk kᵢ Bⱼ

  # define the variables
  k₁,k₂,k₃ = grid.kr,grid.l,grid.m;
  A₁h = @view sol[:,:,:,params.bx_ind]
  A₂h = @view sol[:,:,:,params.by_ind]
  A₃h = @view sol[:,:,:,params.bz_ind]
  B₁  = vars.bx
  B₂  = vars.by
  B₃  = vars.bz

  # Way 2  of appling Curl
  CBᵢh = vars.nonlinh1
  @. CBᵢh = im*(k₂*A₃h - k₃*A₂h)
  ldiv!(B₁, grid.rfftplan, CBᵢh)  

  @. CBᵢh = im*(k₃*A₁h - k₁*A₃h)
  ldiv!(B₂, grid.rfftplan, CBᵢh) 

  @. CBᵢh = im*(k₁*A₂h - k₂*A₁h)
  ldiv!(B₃, grid.rfftplan, CBᵢh)  

  return nothing
end

function UpdateJ!(sol, vars, params, grid)

  # ∇XB = im*( k × B )ₖ = im*ϵ_ijk kᵢ Bⱼ

  # define the variables
  k₁,k₂,k₃ = grid.kr,grid.l,grid.m;
  A₁h = @view sol[:,:,:,params.bx_ind]
  A₂h = @view sol[:,:,:,params.by_ind]
  A₃h = @view sol[:,:,:,params.bz_ind]

  B₁,   B₂,  B₃  =  vars.bx,  vars.by,  vars.bz
  J₁h, J₂h, J₃h  = vars.jxh, vars.jyh, vars.jzh
  @. J₁h *= 0 
  @. J₂h *= 0 
  @. J₃h *= 0

  B₁h = B₂h = B₃h = vars.nonlinh1
  mul!(B₁h, grid.rfftplan, B₁)
  @. J₃h -= im*k₂*B₁h
  @. J₂h += im*k₃*B₁h

  mul!(B₂h, grid.rfftplan, B₂)
  @. J₁h -= im*k₃*B₂h
  @. J₃h += im*k₁*B₂h

  mul!(B₃h, grid.rfftplan, B₃)
  @. J₁h -= im*k₃*B₂h
  @. J₂h += im*k₃*B₁h

  #@. J₁h = im*(k₂*B₃h - k₃*B₂h) #@. J₂h = im*(k₃*B₁h - k₁*B₃h) #@. J₃h = im*(k₁*B₂h - k₂*B₁h)

  return nothing
end

function UpdateΦ!(sol, vars, params, grid)

  # Φ term to conserve the coulomb gauge
  # compute it using 
  # k²*Φh = ∑ᵢ kᵢ( (v×B)ᵢ - ηJᵢ ) 

  # define the variables
  k₁,k₂,k₃ = grid.kr,grid.l,grid.m
  k⁻² = grid.invKrsq
  Φh  = vars.Φh

  @. Φh .= 0
  ε_ijkuⱼbₖ  = vars.nonlin1        
  ε_ijkuⱼbₖh = vars.nonlinh1
  ε_ijkuⱼbₖ  .= 0 
  ε_ijkuⱼbₖh .= 0 
  #compute  ∑ᵢ kᵢ( (v×B)ᵢ - ηJᵢ ) term 
  for (Jᵢh,kᵢ,i) ∈ zip((vars.jxh,vars.jyh,vars.jzh),(k₁,k₂,k₃),(1,2,3))
    # compute the term v × B term using einstein notation
    for (uⱼ,j) ∈ zip((vars.ux,vars.uy,vars.uz),(1,2,3))
      for (bₖ,k) ∈ zip((vars.bx,vars.by,vars.bz),(1,2,3))
        if ϵ(i,j,k) > 0.0
          ϵ_ijk = ϵ(i,j,k)
          @. ε_ijkuⱼbₖ = ϵ_ijk*uⱼ*bₖ
          mul!(ε_ijkuⱼbₖh, grid.rfftplan, ε_ijkuⱼbₖ)
          @. Φh += kᵢ*ε_ijkuⱼbₖh
        end
      end
    end
    # compute η ∇ × B term
    @. Φh += kᵢ*params.η*Jᵢh
  end

  # compute ∑ᵢ kᵢ( (v×B)ᵢ - ηJᵢ )/k²
  @. Φh *= k⁻² 

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
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]));
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]));
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])); 

  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z");

  #Update B Advection
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
  BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z"); 

  return nothing
end

function AMHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Real Conponment
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  UpdateB!(sol, vars, params, grid);

  #Upadte Φ and J in spectral space
  UpdateΦ!(sol, vars, params, grid);
  UpdateJ!(sol, vars, params, grid);

  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z");

  #Update B Advection
  AᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
  AᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
  AᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z"); 

  return nothing
end

end
