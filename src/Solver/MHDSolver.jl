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

# δ function
δ(a::Int,b::Int) = ( a == b ? 1 : 0 );

# checking function of VP method
VP_is_turned_on(params) = hasproperty(params,:U₀x);

function UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="x")

  if direction == "x"

  	# a = {1,2,3} -> {x,y,z} direction
  	a    = 1;
  	kₐ   = grid.kr;
  	k⁻²  = grid.invKrsq;
  	∂uᵢh∂t = @view N[:,:,:,params.ux_ind];

  elseif direction == "y"

  	a    = 2;
  	kₐ   = grid.l;
  	k⁻²  = grid.invKrsq;
  	∂uᵢh∂t = @view N[:,:,:,params.uy_ind];

  elseif direction == "z"
  	a    = 3;
  	kₐ   = grid.m;
  	k⁻²  = grid.invKrsq;
  	∂uᵢh∂t = @view N[:,:,:,params.uz_ind];

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
        @. vars.nonlin1  *= 0;
        @. vars.nonlinh1 *= 0;
        bᵢbⱼ_minus_uᵢuⱼ  = vars.nonlin1;  
        bᵢbⱼ_minus_uᵢuⱼh = vars.nonlinh1;

        # Perform Computation in Real space
        @. bᵢbⱼ_minus_uᵢuⱼ = bᵢ*bⱼ - uᵢ*uⱼ;
        mul!(bᵢbⱼ_minus_uᵢuⱼh, grid.rfftplan, bᵢbⱼ_minus_uᵢuⱼ);

        # Perform the Actual Advection update
        @. ∂uᵢh∂t += im*kᵢ*(δ(a,j)-kₐ*kⱼ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh;
        if i != j  # repeat the calculation for u_ij
          @. ∂uᵢh∂t += im*kⱼ*(δ(a,i)-kₐ*kᵢ*k⁻²)*bᵢbⱼ_minus_uᵢuⱼh;
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
  uᵢh = vars.nonlinh1;
  mul!(uᵢh, grid.rfftplan, uᵢ); 
  @. ∂uᵢh∂t += -grid.Krsq*params.ν*uᵢh;
  
  # hyperdiffusion term
  if params.nν > 1
    @. ∂uᵢh∂t += -grid.Krsq^params.nν*params.ν*uᵢh;
  end

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

  @. ∂Bᵢh∂t*= 0;
  uᵢbⱼ_minus_bᵢuⱼ  = vars.nonlin1;        
  uᵢbⱼ_minus_bᵢuⱼh = vars.nonlinh1;
  #Compute the first term, im ∑_j k_j*(b_iu_j - u_ib_j)
  for (bⱼ,uⱼ,kⱼ,j) ∈ zip((vars.bx,vars.by,vars.bz),(vars.ux,vars.uy,vars.uz),(grid.kr,grid.l,grid.m),(1,2,3))
    if a != j
      # Perform Computation in Real space
      @. uᵢbⱼ_minus_bᵢuⱼ = uᵢ*bⱼ - bᵢ*uⱼ;
      
      mul!(uᵢbⱼ_minus_bᵢuⱼh, grid.rfftplan, uᵢbⱼ_minus_bᵢuⱼ);

      # Perform the Actual Advection update
      @. ∂Bᵢh∂t += im*kⱼ*uᵢbⱼ_minus_bᵢuⱼh;  

    end
  end

  # Updating the solid domain if VP flag is ON
  if VP_is_turned_on(params) 
    VPSolver.VP_BᵢUpdate!(∂Bᵢh∂t, kₐ.*k⁻², a, clock, vars, params, grid)
  end

  #Compute the diffusion term  - ηk^2 B_i
  bᵢh = vars.nonlinh1;
  mul!(bᵢh, grid.rfftplan, bᵢ); 
  @. ∂Bᵢh∂t += -grid.Krsq*params.η*bᵢh;

  # hyperdiffusion term
  if params.nη > 1
    @. ∂Bᵢh∂t += -grid.Krsq^params.nη*params.η*bᵢh;
  end
    
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

end
