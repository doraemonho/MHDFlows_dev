module Shear
# ----------
# Shearing Box Module Ref : The Astrophysical Journal, 928:113 (8pp), 2022 Apr
# ----------

export 
  ShearingReMapping!,
  HD_ShearingAdvection!,
  MHD_ShearingAdvection!,
  Shearing_dealias!

using
  CUDA,
  TimerOutputs

using LinearAlgebra: mul!, ldiv!

include("MHDSolver.jl");
include("HDSolver.jl");
HDUᵢUpdate! = MHDSolver.UᵢUpdate!;
MHDUᵢUpdate! = MHDSolver.UᵢUpdate!;
BᵢUpdate! = MHDSolver.BᵢUpdate!;

function ShearingReMapping!(prob)
  ShearingReMapping!(prob.sol, prob.clock, prob.vars, prob.params);
  return nothing
end

function ShearingReMapping!(sol, clock, vars, params)
    q  = params.usr_params.q;
    Ω  = params.usr_params.Ω;
    Lx,Ly = grid.Lx,grid.Ly;
    τΩ = Lx/Ly/q/Ω;
    
    #advect the shear time
    params.usr_params.τ += clock.dt;
    
    # correct the shear after every shear period
    if τ >= τΩ && clock.step > 1
        FieldReMapping!(sol, clock, vars, params)
        params.usr_params.τ = 0;
    end
    
    kx,ky,kz = grid.kr,grid.l,grid.m;
    k²,k⁻²   = grid.Krsq,grid.invKrsq;
    ky₀      = params.usr_params.ky₀;
    τ        = params.usr_params.τ;
    
    # Construct the new shear
    @. ky  = ky₀ + q*Ω*τ*kx;
    @. k²  = kx^2 + ky^2 + kz^2;
    @. k⁻² = 1/k²;
    @views @. k⁻²[k².== 0] .= 0;
    
    return nothing;
end

function FieldReMapping!!(sol, clock, vars, params)
    q  = params.usr_params.q;
    Ω  = params.usr_params.Ω;
    Lx,Ly = grid.Lx,grid.Ly;
    kx,ky,kz = grid.kr,grid.l,grid.m;
    tmp = params.usr_params.tmp;
    τΩ = Lx/Ly/q/Ω;
    
    dky    = @. floor(q*Ω*τΩ*ky);
    ky_new = @. ky + dky;
    # Change ky_new from (1,nky,1) -> (nky,1,1)
    ky_new = permutedims(ky_new,(2,1,3));
    # boardcasting absdky to (nky,nky,1)
    absdky = @. abs(ky .- ky_new);
    absind = findall(maximum(ky) .>= view(ky_new,:,1) .>= minimum(ky));
    
    #Swap the data from ky -> ky'
    for ind ∈ absind 
        ind_new = argmin(absdky[:,ind]);
        @views @. tmp[ind_new, :, :, :] = sol[ind, :, :, :];
    end
    
    # Copy the data from tmp array to sol
    copyto!(sol,tmp);
    @. tmp*=0;
    
    return nothing;
end

function MHD_ShearingUpdate!(N, sol, t, clock, vars, params, grid)
  U₀xh = params.usr_params.U₀xh;
  U₀hy = params.usr_params.U₀yh;
  U₀x = params.usr_params.U₀x;
  U₀y = params.usr_params.U₀y;
  Ω   = params.usr_params.Ω;
  
  ux_ind,uy_ind = params.ux_ind,params.uy_ind;
  
  
  # V-Field Update
  # We compute the - (U₀ ⋅ ∇ u + u ⋅ ∇ U₀) - 2Ω × u terms
  # ∑ⱼ Uₒⱼkⱼuᵢ + uⱼkⱼUₒᵢ 
  # declare the sketch array
  kⱼuᵢh = Uⱼkⱼuᵢh = kⱼUᵢh = uⱼkⱼUᵢh =  vars.nonlinh1;   
  kⱼuᵢ = Uⱼkⱼuᵢ = kⱼUᵢ = uⱼkⱼUᵢ = vars.nonlin1;        
  for (Uᵢh,uᵢ_ind) ∈ zip([U₀xh,U₀yh],[ux_ind,uy_ind])
    @views   uᵢh = sol[:,:,:,uᵢ_ind]; 
    @views ∂uᵢ∂t =   N[:,:,:,uᵢ_ind];
    for (Uⱼ,uⱼ,kⱼ) ∈ zip([U₀x,U₀y],[vars.ux,vars.uy],[grid.kr,grid.l])
      @. kⱼuᵢh  = im*kⱼ*uᵢh;
      ldiv!(kⱼuᵢ,grid.rfftplan,kⱼuᵢh);
      @. Uⱼkⱼuᵢ = Uⱼ*kⱼuᵢ;
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼuᵢ);
      @. ∂uᵢ∂t -= Uⱼkⱼuᵢh;
      
      @. kⱼUᵢh  = im*kⱼ*Uᵢh;
      ldiv!(kⱼUᵢ,grid.rfftplan,kⱼUᵢh);
      @. uⱼkⱼUᵢ = uⱼ*kⱼUᵢ;
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼuᵢ);
      @. ∂uᵢ∂t -= uⱼkⱼUᵢh;
    end
  end
  # - 2Ω × u
  # - Ω(-uy \hat{x} + ux \hat{y})
  @views @. N[:,:,:,ux_ind] += Ω*sol[:,:,:,uy_ind];
  @views @. N[:,:,:,uy_ind] -= Ω*sol[:,:,:,ux_ind];
  
  # B-Field Update
  #We compute the  (-U₀ ⋅ ∇ B + B ⋅ ∇ U₀) 
  kⱼbᵢh = Uⱼkⱼbᵢh = kⱼUᵢh = bⱼkⱼUᵢh =  vars.nonlinh1;   
  kⱼbᵢ = Uⱼkⱼbᵢ = kⱼUᵢ = bⱼkⱼUᵢ = vars.nonlin1;        
  for (Uᵢh,bᵢ_ind) ∈ zip([U₀xh,U₀yh],[bx_ind,by_ind])
    @views   bᵢh = sol[:,:,:,bᵢ_ind]; 
    @views ∂bᵢ∂t =   N[:,:,:,uᵢ_ind];
    for (Uⱼ,bⱼ,kⱼ) ∈ zip([U₀x,U₀y],[vars.bx,vars.by],[grid.kr,grid.l])
      @. kⱼbᵢh  = im*kⱼ*bᵢh;
      ldiv!(kⱼbᵢ,grid.rfftplan,kⱼbᵢh);
      @. uⱼkⱼbᵢ = Uⱼ*kⱼbᵢ;
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼuᵢ);
      @. ∂bᵢ∂t -= Uⱼkⱼbᵢh;
      
      @. kⱼUᵢh  = im*kⱼ*Uᵢh;
      ldiv!(kⱼUᵢ,grid.rfftplan,kⱼUᵢh);
      @. bⱼkⱼUᵢ = bⱼ*kⱼUᵢ;
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼuᵢ);
      @. ∂bᵢ∂t += bⱼkⱼUᵢh;
    end
  end
  
  return nothing;
end

function HD_ShearingUpdate!(N, sol, t, clock, vars, params, grid)
  U₀xh = params.usr_params.U₀xh;
  U₀hy = params.usr_params.U₀yh;
  U₀x = params.usr_params.U₀x;
  U₀y = params.usr_params.U₀y;
  Ω   = params.usr_params.Ω;
  
  ux_ind,uy_ind = params.ux_ind,params.uy_ind;
  
  
  # V-Field Update
  # We compute the - (U₀ ⋅ ∇ u + u ⋅ ∇ U₀) - 2Ω × u terms
  # ∑ⱼ Uₒⱼkⱼuᵢ + uⱼkⱼUₒᵢ 
  # declare the sketch array
  kⱼuᵢh = Uⱼkⱼuᵢh = kⱼUᵢh = uⱼkⱼUᵢh =  vars.nonlinh1;   
  kⱼuᵢ = Uⱼkⱼuᵢ = kⱼUᵢ = uⱼkⱼUᵢ = vars.nonlin1;        
  for (Uᵢh,uᵢ_ind) ∈ zip([U₀xh,U₀yh],[ux_ind,uy_ind])
    @views   uᵢh = sol[:,:,:,uᵢ_ind]; 
    @views ∂uᵢ∂t =   N[:,:,:,uᵢ_ind];
    for (Uⱼ,uⱼ,kⱼ) ∈ zip([U₀x,U₀y],[vars.ux,vars.uy],[grid.kr,grid.l])
      @. kⱼuᵢh  = im*kⱼ*uᵢh;
      ldiv!(kⱼuᵢ,grid.rfftplan,kⱼuᵢh);
      @. Uⱼkⱼuᵢ = Uⱼ*kⱼuᵢ;
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼuᵢ);
      @. ∂uᵢ∂t -= Uⱼkⱼuᵢh;
      
      @. kⱼUᵢh  = im*kⱼ*Uᵢh;
      ldiv!(kⱼUᵢ,grid.rfftplan,kⱼUᵢh);
      @. uⱼkⱼUᵢ = uⱼ*kⱼUᵢ;
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼuᵢ);
      @. ∂uᵢ∂t -= uⱼkⱼUᵢh;
    end
  end
  # - 2Ω × u
  # - Ω(-uy \hat{x} + ux \hat{y})
  @views @. N[:,:,:,ux_ind] += Ω*sol[:,:,:,uy_ind];
  @views @. N[:,:,:,uy_ind] -= Ω*sol[:,:,:,ux_ind];
  
  return nothing;
end

function MHD_ShearingAdvection!(N, sol, t, clock, vars, params, grid)
    
  #Update V + B Real Conponment
  @timeit_debug params.debugTimer "FFT Update" begin
    ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
    ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
    ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
    ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]));
    ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]));
    ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])); 
  end
  #Update V Advection
  @timeit_debug params.debugTimer "UᵢUpdate" begin
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z");
  end
  #Update B Advection
  @timeit_debug params.debugTimer "BᵢUpdate" begin
    BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
    BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
    BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z"); 
  end

  @timeit_debug params.debugTimer "ShearingUpdate" begin
  MHD_ShearingUpdate!(N, sol, t, clock, vars, params, grid);
  end
  return nothing
end

function HD_ShearingAdvection!(N, sol, t, clock, vars, params, grid)
    
  #Update V + B Real Conponment
  @timeit_debug params.debugTimer "FFT Update" begin
    ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
    ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
    ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  end
  #Update V Advection
  @timeit_debug params.debugTimer "UᵢUpdate" begin
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z");
  end

  @timeit_debug params.debugTimer "ShearingUpdate" begin
  ShearingUpdate!(N, sol, t, clock, vars, params, grid);
  end
  return nothing
end

function Shearing_dealias!(fh, grid)
    @assert grid.nkr == size(fh)[1]
    # kfilter = 2/3*kmax
    kfilter = 2/3*3*grid.nkr^2;
    @views @. fh[grid.Krsq.>=kfilter,:] = 0;
    return nothing;
end

end