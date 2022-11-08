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

function UᵢUpdate!(N, sol, t, clock, vars, params, grid;direction = "x")
  if direction == "x"

  # a = {1,2,3} -> {x,y,z} direction
  a    = 1;
  kₐ   = grid.kr;
  k⁻²  = grid.invKrsq;
  ∂pᵢh∂t = @view N[:,:,:,params.px_ind];
elseif direction == "y"
  a    = 2;
  kₐ   = grid.l;
  k⁻²  = grid.invKrsq;
  ∂pᵢh∂t = @view N[:,:,:,params.py_ind];
elseif direction == "z"
  a    = 3;
  kₐ   = grid.m;
  k⁻²  = grid.invKrsq;
  ∂pᵢh∂t = @view N[:,:,:,params.pz_ind];
else
  error("Warning : Unknown direction is declerad")
end

end

function MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)

  #Update ρ + P + V + B Real Conponment
  ldiv!(vars.ρ , grid.rfftplan, deepcopy(@view sol[:, :, :, params.ρ_ind ]));
  ldiv!(vars.P , grid.rfftplan, deepcopy(@view sol[:, :, :, params.P_ind ]));  
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