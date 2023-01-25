module Shear
# ----------
# Shearing Box Module Ref : The Astrophysical Journal, 928:113 (8pp), 2022 Apr
# ----------

#
#Note: Haven't finish, check the name vars for the loops.
#
#
#
#
export 
  Shearing_coordinate_update!,
  Shearing_remapping!,
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

function Shearing_remapping!(prob)
  Shearing_remapping!(prob.sol, prob.clock, prob.vars, prob.params, prob.grid);
  return nothing
end

function Shearing_coordinate_update!(N, sol, t, clock, vars, params, grid)
    q  = params.usr_params.q;
    Ω  = params.usr_params.Ω;
    τΩ = params.usr_params.τΩ;
    Lx,Ly = grid.Lx,grid.Ly;
    
    #Shear time intreval in sub-time-step
    dτ = clock.t - t;
    
    kx,ky,kz = grid.kr,grid.l,grid.m;
    k²,k⁻²   = grid.Krsq,grid.invKrsq;
    ky₀      = params.usr_params.ky₀;
    τ        = params.usr_params.τ;
    
    # Construct the new shear coordinate
    @. ky  = ky₀ + q*Ω*(τ+dτ)*kx;
    @. k²  = kx^2 + ky^2 + kz^2;
    @. k⁻² = 1/k²;
    @views @. k⁻²[k².== 0] .= 0;

    return nothing
end

function Shearing_remapping!(sol, clock, vars, params, grid)
    q  = params.usr_params.q;
    Ω  = params.usr_params.Ω;
    τΩ = params.usr_params.τΩ;
    Lx,Ly = grid.Lx,grid.Ly;
    
    #advect the shear time
    params.usr_params.τ += clock.dt;
    
    # correct the shear after every shear period
    if τ >= τΩ && clock.step > 1
        Field_remapping!(sol, clock, vars, params, grid)
        params.usr_params.τ = 0;
    end
    
    kx,ky,kz = grid.kr,grid.l,grid.m;
    k²,k⁻²   = grid.Krsq,grid.invKrsq;
    ky₀      = params.usr_params.ky₀;
    τ        = params.usr_params.τ;
    
    # Construct the new shear coordinate
    @. ky  = ky₀ + q*Ω*τ*kx;
    @. k²  = kx^2 + ky^2 + kz^2;
    @. k⁻² = 1/k²;
    @views @. k⁻²[k².== 0] .= 0;
    
    return nothing;
end

function Field_remapping_old!(sol, clock, vars, params)
  q  = params.usr_params.q;
  Ω  = params.usr_params.Ω;
  τΩ = params.usr_params.τΩ;
  kx,ky,kz = grid.kr,grid.l,grid.m;
  tmp = params.usr_params.tmp;

  dky    = @. floor(q*Ω*τΩ*kx); #size = (nkr, 1,1)
  ky_new = @. ky + dky;         #size = (nkr,nl,1)
  # Change ky_new from (1,nky,1) -> (nky,1,1) < ---
  #ky_new = permutedims(ky_new,(2,1,3));      < --- 
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

function Field_remapping!(sol, clock, vars, params, grid)
  T  = eltype(grid)
  q  = params.usr_params.q;
  Ω  = params.usr_params.Ω;
  τΩ = params.usr_params.τΩ;
  kx,ky0 = grid.kr, grid.l1D
  tmp = params.usr_params.tmp;
  
  # Set up of CUDA threads & block
  threads = ( 32, 8, 1) #(9,9,9)
  blocks  = ( ceil(Int,size(sol,1)/threads[1]), ceil(Int,size(sol,2)/threads[2]), ceil(Int,size(sol,3)/threads[3]))  
  Nfield  = size(sol,4)

  for n = 1:Nfield
    fieldᵢ = (@view sol[:,:,:,n])::CuArray{T,3} 
      tmpᵢ = (@view tmp[:,:,:,n])::CuArray{T,3}
    @cuda blocks = blocks threads = threads Field_remapping_CUDA!(tmpᵢ, fieldᵢ,
                                                                  q, Ω, τΩ, kx, ky0)
  end

  # Copy the data from tmp array to sol
  copyto!(sol,tmp);
  @. tmp*=0;
    
  return nothing;
end


function Field_remapping_CUDA!(tmpᵢ, fieldᵢ,
                               q, Ω, τΩ, kx, ky0)
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
  nx,ny,nz = size(fieldᵢ)

  if k ∈ (1:nz) && j ∈ (1:ny) && i ∈ (1:nx)
    dky   = floor(q*Ω*τΩ*kx[i])
    kynew = ky0[j] + dky
    if  maxval(ky0) >= kynew >= minval(ky0)
      jnew = minloc(abs(ky0 - kynew), 1)
      tmpᵢ[i,jnew,k] = fieldᵢ[i,j,k]
    end
  end
  return nothing
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
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼUᵢ);
      @. ∂bᵢ∂t += bⱼkⱼUᵢh;
    end
  end
  
  return nothing;
end

function HD_ShearingUpdate!(N, sol, t, clock, vars, params, grid)
  U₀xh = params.usr_params.U₀xh;
  U₀hy = params.usr_params.U₀yh;
  U₀x  = params.usr_params.U₀x;
  U₀y  = params.usr_params.U₀y;
  Ω    = params.usr_params.Ω;
  
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
      mul!(Uⱼkⱼuᵢh,grid.rfftplan,uⱼkⱼUᵢ);
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
  @timeit_debug params.debugTimer "FFT Update" CUDA.@sync begin
    ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
    ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
    ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
    ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]));
    ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]));
    ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])); 
  end
  #Update V Advection
  @timeit_debug params.debugTimer "UᵢUpdate" CUDA.@sync begin
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z");
  end
  #Update B Advection
  @timeit_debug params.debugTimer "BᵢUpdate" CUDA.@sync begin
    BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
    BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
    BᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z"); 
  end

  @timeit_debug params.debugTimer "ShearingUpdate" CUDA.@sync begin
  MHD_ShearingUpdate!(N, sol, t, clock, vars, params, grid);
  end
  return nothing
end

function HD_ShearingAdvection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Real Conponment
  @timeit_debug params.debugTimer "FFT Update" CUDA.@sync begin
    ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
    ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
    ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  end
  #Update V Advection
  @timeit_debug params.debugTimer "UᵢUpdate" CUDA.@sync begin
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x");
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y");
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z");
  end

  @timeit_debug params.debugTimer "ShearingUpdate" CUDA.@sync begin
  HD_ShearingUpdate!(N, sol, t, clock, vars, params, grid);
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