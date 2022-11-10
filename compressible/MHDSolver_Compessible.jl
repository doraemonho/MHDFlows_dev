module MHDSolver_compressible
# ----------
# Compessible Navier–Stokes Solver for 3D Magnetohydrodynamics Problem
# ----------

function ρUpdate!(N, sol, t, clock, vars, params, grid)

  ∂ρ∂t = @view   N[:,:,:,params.ρ_ind];
     ρ = @view sol[:,:,:,params.ρ_ind];

# Solving the continuity equation
# ∂ρ∂t = -∇· (ρv) => ∑_i -im*kᵢ(ρvᵢ)ₕ
  for (uᵢ,kᵢ) ∈ zip([vars.ux,vars.uy,vars.uz],[grid.kr,grid.l,grid.m])
      # Initialization of sketch array
      @. vars.nonlin1  *= 0;
      @. vars.nonlinh1 *= 0;
      ρvᵢ  = vars.nonlin1;
      ρvᵢh = vars.nonlin1h;
      # Perform Computation in Real space
      @. ρvᵢ = ρ*uᵢ;
      mul!(ρvᵢh, grid.rfftplan, ρvᵢ);
      # Perform the Actual Advection update
      @. ∂ρ∂t = -im*kᵢ* ρvᵢh;
  end
  return nothing;
end

# fft  - space parameters -> ρ px py pz bx by bz 
# real - space parameters -> ρ ux uy uz bx by bz
function UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction = "x")
  ν = params.ν;
  cₛ = params.cₛ;
  if direction == "x"
    # a = {1,2,3} -> {x,y,z} direction
    kᵢ   = grid.kr;
    uᵢ   = vars.ux;
    bᵢ   = vars.bx;
    ∂pᵢh∂t = @view N[:,:,:,params.px_ind];
  elseif direction == "y"
    kᵢ   = grid.l;
    uᵢ   = vars.uy;
    bᵢ   = vars.by;
    ∂pᵢh∂t = @view N[:,:,:,params.py_ind];
  elseif direction == "z"
    kᵢ   = grid.m;
    uᵢ   = vars.uz;
    bᵢ   = vars.bz;
    ∂pᵢh∂t = @view N[:,:,:,params.pz_ind];
  end

  @. ∂pᵢh∂t*=0;
  #momentum and magnetic field part
  bᵢbⱼ_minus_ρuᵢuⱼ  = vars.nonlin1;  
  bᵢbⱼ_minus_ρuᵢuⱼh = vars.nonlinh1;
  for (uⱼ,bⱼ,kⱼ) ∈ zip([vars.ux,vars.uy,vars.uz],[vars.bx,vars.by,vars.bz],[grid.kr,grid.l,grid.m])
    # pseudo part
    @. bᵢbⱼ_minus_ρuᵢuⱼ  = bᵢ*bⱼ- ρ*uᵢ*uⱼ;
    # spectral part
    mul!(bᵢbⱼ_minus_ρuᵢuⱼh, grid.rfftplan, bᵢbⱼ_minus_ρuᵢuⱼ);
    @. ∂pᵢh∂t += im*kⱼ*bᵢbⱼ_minus_ρuᵢuⱼh;
  end

  #pressure part
  P_tot  = vars.nonlin1;  
  P_toth = vars.nonlinh1;
  @. P_tot= ρ*cₛ^2 + vars.bx^2 + vars.by^2 + vars.bz^2
  mul!(P_toth, grid.rfftplan, P_tot);
  @. ∂pᵢh∂t -= im*kᵢ*P_toth;

  # viscosity part
  ρSᵢⱼ  = vars.nonlin1;  
  ρSᵢⱼh = vars.nonlinh1;
  for (uⱼ,bⱼ,kⱼ) ∈ zip([vars.ux,vars.uy,vars.uz],[vars.bx,vars.by,vars.bz],[grid.kr,grid.l,grid.m])
    @. ρSᵢⱼ = ρ*Sᵢⱼ;
    mul!(ρS_ijh, grid.rfftplan, ρSᵢⱼ);
    @. ∂pᵢh∂t -= kⱼ*2*ν*ρSᵢⱼh;
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