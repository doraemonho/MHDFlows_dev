#-----------------------
# Negative damping turbulence module:
# We consider the following force Fᵢ for direction i :
# Fᵢ = fᵢ*uᵢ
#----------------------

mutable struct ND_vars{Aphys,Atrans}
  fx  :: Aphys
  fy  :: Aphys
  fz  :: Aphys
end

function SetUpND!(prob,fx_,fy_,fz_; F0 = 1, kf = 2)
  grid = prob.grid;
  vars = prob.vars;
  x,y,z = grid.x,grid.y,grid.z;
  nx,ny,nz = grid.nx,grid.ny,grid.nz;
  fx,fy,fz = vars.usr_vars.fx , vars.usr_vars.fy, vars.usr_vars.fz;
  copyto!(fx,fx_);
  copyto!(fy,fy_);
  copyto!(fy,fz_);
  return nothing;
end

function NDForceDriving!(N, sol, t, clock, vars, params, grid)
  uvars = vars.usr_vars; 
  Fᵢ  = vars.nonlin1;
  Fᵢh = vars.nonlinh1;
  for (fᵢ,uᵢ,uᵢind) in zip([uvars.fx,uvars.fy,uvars.fz],
                           [vars.ux,vars.uy,vars.uz],
                           [params.ux_ind,params.uy_ind,params.uz_ind])

    @. Fᵢh*=0;
    @. Fᵢ = fᵢ*uᵢ;
    mul!(Fᵢh, grid.rfftplan, Fᵢ);
    @views @. N[:,:,:,uᵢind] += Fᵢh;
  end
  return nothing;
end

function GetNDvars_And_function(::Dev, nx::Int,ny::Int,nz::Int; T = Float32) where Dev
  @devzeros Dev T  ( nx, ny, nz) fx  fy fz
    
  return  ND_vars(fx,fy,fxh,fyh), NDForceDriving!;  
end