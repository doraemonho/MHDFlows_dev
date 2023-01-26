# ----------
# TimeStepper for Shearing Box Simulation
# ----------
struct eSSPIFRK3TimeStepper{T,TL} <: FourierFlows.AbstractTimeStepper{T}
  L₀ :: TL
  L₁ :: TL
  L₂ :: TL
  L₃ :: TL
  u₀ :: T
  u₁ :: T
  u₂ :: T
  N₀ :: T
  N₁ :: T
  N₂ :: T
end

function eSSPIFRK3TimeStepper(equation, dev::Device=CPU())
  @devzeros typeof(dev) equation.T equation.dims          u₀ u₁ u₂ N₀ N₁ N₂
  @devzeros typeof(dev) equation.T equation.dims[1:end-1] L₀ L₁ L₂ L₃
  
  return eSSPIFRK3TimeStepper(L₀, L₁, L₂, L₃, u₀, u₁, u₂, N₀, N₁, N₂)
end

function getL!(Lᵢ, t, clock, params, grid)
  q  = params.usr_params.q;
  ν  = params.usr_params.ν;
  
  kx,ky,kz = grid.kr,grid.l,grid.m
  ky₀      = params.usr_params.ky₀
  k2xz     = params.usr_params.k2xz
  Krsq     = grid.Krsq
  dt       = t - clock.t 
  τ        = params.usr_params.τ + dt
  
  @. ky   = ky₀ + q*τ*kx
  @. k2xz = kx^2 + kz^2
  @. Lᵢ   = -ν*(k2xz^2*ky + 2/3*k2xz*ky^3 + ky^5/5)/(q*kx)
  @. Lᵢ   = Lᵢ/maximum(Krsq)^2
  @. @views Lᵢ[1,:,:] = 0
  return nothing
end

stepforward!(prob) =
  stepforward!(prob.sol, prob.clock, prob.timestepper, prob.eqn, prob.vars, prob.params, prob.grid)

function stepforward!(sol, clock, ts::eSSPIFRK3TimeStepper, equation, vars, params, grid)
  eSSPIFRK3substeps!(sol, clock, ts, equation, vars, params, grid)
  clock.t += clock.dt
  clock.step += 1
  return nothing
end

# L\_i has to change
function eSSPIFRK3substeps!(sol, clock, ts, equation, vars, params, grid)

  dt = clock.dt
  t  = clock.t
  getL!(ts.L₀, t         , clock, params, grid)
  getL!(ts.L₁, t + 2/3*dt, clock, params, grid)
  getL!(ts.L₂, t + 2/3*dt, clock, params, grid)
  getL!(ts.L₃, t +     dt, clock, params, grid)
  
  # Substep 1
  copyto!(ts.u₀, sol)
  equation.calcN!(ts.N₀, sol, clock.t, clock, vars, params, grid)
  eSSPIFRK3_step1!(ts.u₁, ts.u₀, ts.N₀, ts.L₀, ts.L₁, dt)
  
  # Substep 2
  t2 = clock.t + clock.dt*2/3
  equation.calcN!(ts.N₁, ts.u₁, t2, clock, vars, params, grid)
  eSSPIFRK3_step2!(ts.u₂, ts.u₁, ts.u₀, ts.N₁, ts.L₀, ts.L₂, dt)
  
  # Substep 3
  t3 = clock.t + clock.dt
  equation.calcN!(ts.N₂, ts.u₂, t2, clock, vars, params, grid)
  eSSPIFRK3_step3!(sol, ts.u₀, ts.u₂, ts.N₀, ts.N₂, ts.L₀, ts.L₂, ts.L₃, dt)
  
  return nothing
end

function eSSPIFRK3_step1!(u1::CuArray{Complex{T},4}, u0, N0, L0, L1, dt) where T
  nf = size(u1)[end]
  for i = 1:nf
  eSSPIFRK3_step1!( view(u1,:,:,:,i), view(u0,:,:,:,i), view(N0,:,:,:,i),
                    L0, L1, dt) 
  end
end

function eSSPIFRK3_step2!(u2::CuArray{Complex{T},4}, u1, u0, N1, L0, L2, dt) where T
  nf = size(u1)[end]
  for i = 1:nf
  eSSPIFRK3_step2!(view(u2,:,:,:,i), view(u1,:,:,:,i), view(u0,:,:,:,i), view(N1,:,:,:,i),
                   L0, L2, dt)
  end
end

function eSSPIFRK3_step3!(sol::CuArray{Complex{T},4}, u2, u1, u0, L0, L2, N1, dt) where T
  nf = size(u1)[end]
  for i = 1:nf
  eSSPIFRK3_step3!(view(sol,:,:,:,i), view(u0,:,:,:,i), view(u2,:,:,i), view(N0,:,:,:,i), view(N2,:,:,:,i), 
                   L0, L2, L3, dt)
  end
end

function eSSPIFRK3_step1!(u1, u0, N0, L0, L1, dt)
  @. u1 = exp(L1 - L0)*(u0 + 2/3*dt*N0)
  return nothing
end

function eSSPIFRK3_step2!(u2, u1, u0, N1, L0, L2, dt)
  @. u2 = 2/3*exp(L2 - L0)*u0 + 1/3*u1 + 4/9*dt*N1
  return nothing
end

function eSSPIFRK3_step3!(sol, u0, u2, N0, N2, L0, L2, L3, dt)
  #u0 u1 u2 are the sub-step
  @. sol = exp(L3 - L0)*(37/64*u0 + 5/32*dt*N0) + exp(L3 - L2)*(27/64*u2 + 9/16*dt*N2)
  return nothing
end