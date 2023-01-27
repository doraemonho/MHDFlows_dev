module Shear
# ----------
# Shearing Box Module Ref : The Astrophysical Journal, 928:113 (8pp), 2022 Apr
# ----------

#
#Note: Haven't finish, check the name vars for the loops.
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

include("MHDSolver.jl")
include("HDSolver.jl")
HDUᵢUpdate! = MHDSolver.UᵢUpdate!
MHDUᵢUpdate! = MHDSolver.UᵢUpdate!
BᵢUpdate! = MHDSolver.BᵢUpdate!

function Shearing_remapping!(prob)
  Shearing_remapping!(prob.sol, prob.clock, prob.vars, prob.params, prob.grid)
  return nothing
end

function Shearing_coordinate_update!(N, sol, t, clock, vars, params, grid)
  q  = params.usr_params.q
  τΩ = params.usr_params.τΩ
  Lx,Ly = grid.Lx,grid.Ly
  
  #Shear time interval in sub-time-step
  dτ = clock.t - t
  
  kx,ky,kz = grid.kr,grid.l,grid.m
  k²,k⁻²   = grid.Krsq,grid.invKrsq
  ky₀      = params.usr_params.ky₀
  τ        = params.usr_params.τ
  
  # Construct the new shear coordinate
  @. ky  = ky₀ + q*(τ+dτ)*kx
  @. k²  = kx^2 + ky^2 + kz^2
  @. k⁻² = 1/k²
  @views @. k⁻²[k².== 0] .= 0

  return nothing
end

function Shearing_remapping!(sol, clock, vars, params, grid)
  t  = clock.t
  q  = params.usr_params.q
  τΩ = params.usr_params.τΩ
  Lx,Ly = grid.Lx,grid.Ly
  
  #increase the shear time
  params.usr_params.τ += clock.dt
  
  # correct the shear after every shear period
  if params.usr_params.τ >= τΩ && clock.step > 1
    Field_remapping!(sol, clock, vars, params, grid)
    params.usr_params.τ = 0
  end   
  return nothing
end

function Field_remapping!(sol, clock, vars, params, grid)
  T  = eltype(grid)
  q  = params.usr_params.q
  τΩ = params.usr_params.τΩ
  kx,ky0 = grid.kr, grid.l1D
  Lx     = grid.Lx
  tmp = params.usr_params.tmp
    
  kymin,kymax = minimum(ky0),maximum(ky0)
  
  # Set up of CUDA threads & block
  threads = ( 32, 8, 1) #(9,9,9)
  blocks  = ( ceil(Int,size(sol,1)/threads[1]), ceil(Int,size(sol,2)/threads[2]), ceil(Int,size(sol,3)/threads[3]))  
  Nfield  = size(sol,4)

  for n = 1:Nfield
    fieldᵢ = (@view sol[:,:,:,n])::CuArray{Complex{T},3} 
      tmpᵢ = (@view tmp[:,:,:,n])::CuArray{Complex{T},3}
    @cuda blocks = blocks threads = threads Field_remapping_CUDA!(tmpᵢ, fieldᵢ,
                                                                  q, τΩ, kx, ky0, kymin, kymax, Lx)
  end

  # Copy the data from tmp array to sol
  copyto!(sol,tmp)
  @. tmp*=0
    
  return nothing
end

function MHD_ShearingUpdate!(N, sol, t, clock, vars, params, grid)
  U₀xh = params.usr_params.U₀xh
  U₀yh = params.usr_params.U₀yh
  U₀x  = params.usr_params.U₀x
  U₀y  = params.usr_params.U₀y
  q    = params.usr_params.q
  
  ux_ind,uy_ind = params.ux_ind,params.uy_ind
  bx_ind,by_ind = params.bx_ind,params.by_ind
#    exp_terms(iux) = nl(iux) + fux - zi*kxt*p + 2.d0*shear_flg*uy
#    exp_terms(iuy) = nl(iuy) + fuy - zi*ky *p - (2.d0 - q)*shear_flg*ux  
#    exp_terms(iby) = nl(iby) - q*shear_flg*bx
  #@. N[:,:,:,ux_ind] +=  -(2 - q)*N[:,:,:,uy_ind]
  #@. N[:,:,:,uy_ind] +=  +   2   *N[:,:,:,ux_ind]
  #@. N[:,:,:,bx_ind] +=  -   q   *N[:,:,:,by_ind]
  return nothing
end

function Field_remapping_CUDA!(tmpᵢ, fieldᵢ,
                               q, τΩ, kx, ky0, kymin, kymax, Lx)
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
  nx,ny,nz = size(fieldᵢ)
  nky = length(ky0)
  if k ∈ (1:nz) && j ∈ (1:ny) && i ∈ (1:nx)
    dky   = floor(q*τΩ*kx[i])
    kynew = ky0[j] + dky
    if  kymax >= kynew >= kymin
      mindky = abs(ky0[1] - kynew)   
      jnew   = 1      
      for kk = 2:nky
        if mindky > abs(ky0[kk] - kynew)
          jnew = kk  
          mindky = abs(ky0[kk] - kynew)          
        end
      end
      tmpᵢ[i,jnew,k] = fieldᵢ[i,j,k]
    end
  end
  return nothing
end

function MHD_ShearingAdvection!(N, sol, t, clock, vars, params, grid)
    
  #Update V + B Real Conponment
  #@timeit_debug params.debugTimer "FFT Update" CUDA.@sync begin
    ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]))
    ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]))
    ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]))
    ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
    ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
    ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 
  #end
  #Update V Advection
  #@timeit_debug params.debugTimer "UᵢUpdate" CUDA.@sync begin
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="x")
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="y")
    MHDUᵢUpdate!(N, sol, t, clock, vars, params, grid;direction="z")
  #end
  #Update B Advection
  #@timeit_debug params.debugTimer "BᵢUpdate" CUDA.@sync begin
    BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="x")
    BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="y")
    BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="z") 
  #end

  #@timeit_debug params.debugTimer "ShearingUpdate" CUDA.@sync begin
    MHD_ShearingUpdate!(N, sol, t, clock, vars, params, grid)
  #end
  return nothing
end

function HD_ShearingAdvection!(N, sol, t, clock, vars, params, grid)

  #Update V + B Real Conponment
  @timeit_debug params.debugTimer "FFT Update" CUDA.@sync begin
    ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]))
    ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]))
    ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]))
  end
  #Update V Advection
  @timeit_debug params.debugTimer "UᵢUpdate" CUDA.@sync begin
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="x")
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="y")
    HDUᵢUpdate!(N, sol, t, clock, vars, params, grid; direction="z")
  end

  #@timeit_debug params.debugTimer "ShearingUpdate" CUDA.@sync begin
    HD_ShearingUpdate!(N, sol, t, clock, vars, params, grid)
  #end
  return nothing
end

function Shearing_dealias!(fh, grid)
  @assert grid.nkr == size(fh)[1]
  # kfilter = 2/3*kmax
  kfilter = (1/2*grid.nl)^2
  #@views @. fh[grid.Krsq.>=kfilter,:,:,:] = 0

  # Set up of CUDA threads & block
  threads = ( 32, 8, 1)
  blocks  = ( ceil(Int,size(fh,1)/threads[1]), ceil(Int,size(fh,2)/threads[2]), ceil(Int,size(fh,3)/threads[3]))  
  Nfield  = size(fh,4)

  for n = 1:Nfield
    fhᵢ = (@view fh[:,:,:,n])
    @cuda blocks = blocks threads = threads Shearing_dealias_CUDA!(fhᵢ, grid.Krsq, kfilter)
  end
  return nothing
end

function Shearing_dealias_CUDA!(fh, Krsq, kfilter)
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
  nx,ny,nz = size(fh)
  if k ∈ (1:nz) && j ∈ (1:ny) && i ∈ (1:nx)
    if Krsq[i,j,k] >= kfilter
      fh[i,j,k] = 0.0
    end
  end
  return nothing
end

end