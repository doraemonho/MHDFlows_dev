module MHDSolver_compressible
# ----------
# Compessible Navier–Stokes Solver for 3D Magnetohydrodynamics Problem
# ----------

export MHDcalcN_advection!

include("MHDSolver.jl");
include("HDSolver_Compessible.jl")
ρUpdate! = HDSolver_compressible.ρUpdate!;
BᵢUpdate! = MHDSolver.BᵢUpdate!;

using LinearAlgebra: mul!, ldiv!

# Definition of physical parameter between real space and spectral sapce
# fft  - space parameters -> ρ px py pz bx by bz 
# real - space parameters -> ρ ux uy uz bx by bz

# Solving the momentum equation
# ∂pᵢ∂t + ∑ⱼ ∂/∂xⱼ( ρ*uᵢuⱼ + δᵢⱼP_tot - bᵢbⱼ - 2νρSᵢⱼ)) = ρFᵢ
function UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction = "x")
  ν = params.ν;
  cₛ = params.cₛ;
  k₁,k₂,k₃    = grid.kr,grid.l,grid.m;
  u₁h,u₂h,u₃h = vars.uxh,vars.uyh,vars.uzh;
  if direction == "x"
    # i = {1,2,3} -> {x,y,z} direction
    a    = 1;
    kᵢ   = grid.kr;
    uᵢ   = vars.ux;
    uᵢh  = vars.uxh;
    bᵢ   = vars.bx;
    ∂pᵢh∂t = @view N[:,:,:,params.ux_ind];
  elseif direction == "y"
    a    = 2;
    kᵢ   = grid.l;
    uᵢ   = vars.uy;
    uᵢh  = vars.uyh;
    bᵢ   = vars.by;
    ∂pᵢh∂t = @view N[:,:,:,params.uy_ind];
  elseif direction == "z"
    a    = 3;
    kᵢ   = grid.m;
    uᵢ   = vars.uz;
    uᵢh  = vars.uzh;
    bᵢ   = vars.bz;
    ∂pᵢh∂t = @view N[:,:,:,params.uz_ind];
  end

  @. ∂pᵢh∂t*=0;
  ρ = vars.ρ;
  #momentum and magnetic field part
  bᵢbⱼ_minus_ρuᵢuⱼ  = vars.nonlin1;  
  bᵢbⱼ_minus_ρuᵢuⱼh = vars.nonlinh1;
  for (uⱼ,bⱼ,kⱼ) ∈ zip((vars.ux,vars.uy,vars.uz),(vars.bx,vars.by,vars.bz),(grid.kr,grid.l,grid.m))
    # pseudo part
    @. bᵢbⱼ_minus_ρuᵢuⱼ  = bᵢ*bⱼ - ρ*uᵢ*uⱼ;
    # spectral part
    mul!(bᵢbⱼ_minus_ρuᵢuⱼh, grid.rfftplan, bᵢbⱼ_minus_ρuᵢuⱼ);
    @. ∂pᵢh∂t += im*kⱼ*bᵢbⱼ_minus_ρuᵢuⱼh;
  end

  # pressure part
  P_tot  = vars.nonlin1;  
  P_toth = vars.nonlinh1;
  @. P_tot= ρ*cₛ^2 + vars.bx^2 + vars.by^2 + vars.bz^2
  mul!(P_toth, grid.rfftplan, P_tot);
  @. ∂pᵢh∂t -= im*kᵢ*P_toth;

  # viscosity part
  Sᵢⱼ ,ρSᵢⱼ  = vars.nonlin1 ,vars.nonlin2;  
  Sᵢⱼh,ρSᵢⱼh = vars.nonlinh1,vars.nonlinh2;
  for (uⱼh,kⱼ,j) ∈ zip((vars.uxh,vars.uyh,vars.uzh),(grid.kr,grid.l,grid.m),(1,2,3))
    if a == j
      @. Sᵢⱼh += im*(kᵢ*uᵢh - (k₁*u₁h + k₂*u₂h + k₃*u₃h)*0.3333333333);
    else
      @. Sᵢⱼh += 0.5*im*(kᵢ*uⱼh + kⱼ*uᵢh);
    end
    ldiv!(Sᵢⱼ, grid.rfftplan, Sᵢⱼh);
    @. ρSᵢⱼ = ρ*Sᵢⱼ;
    mul!(ρSᵢⱼh, grid.rfftplan, ρSᵢⱼ);
    @. ∂pᵢh∂t += im*kⱼ*2*ν*ρSᵢⱼh;
  end

  return nothing;
end

function MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update ρ + P + V + B Real Conponment
  ldiv!(vars.ρ , grid.rfftplan, deepcopy(@view sol[:, :, :, params.ρ_ind ]));
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]));
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]));
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind]));

  #Update momentum back to velocity
  @. vars.ux/=vars.ρ;
  @. vars.uy/=vars.ρ;
  @. vars.uz/=vars.ρ;
  
  #Copy the spectral conponment to sketch array
  mul!(vars.uxh, grid.rfftplan, vars.ux);
  mul!(vars.uyh, grid.rfftplan, vars.uy);
  mul!(vars.uzh, grid.rfftplan, vars.uz);
  
  #Update continuity equation
  ρUpdate!(N, sol, t, clock, vars, params, grid);
  
  #Update V Advection
  UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="x");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="y");
  UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="z");
  
  #Update B Advection
  BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="x");
  BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="y");
  BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="z"); 
  
  return nothing;
end

end