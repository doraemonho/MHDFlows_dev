# ----------
# Implicit timeStepper for EMHD simulation
# note: while the module is called HM89, but no longer related to HM89....
# ----------

struct HM89TimeStepper{T,TL} <: FourierFlows.AbstractTimeStepper{T}
  F₀    :: T
  F₁    :: T
  Bₘ₋₁  :: T
  Bₘ    :: T
  Bₘ₊₁  :: T
  B₀    :: T
  gₘ    :: T
  gₘ₋₁  :: T
  c     :: TL
end

function HM89TimeStepper(equation, dev::Device=CPU())
  @devzeros typeof(dev) equation.T equation.dims  F₀ F₁ Bₘ₋₁ Bₘ Bₘ₊₁ B₀ gₘ gₘ₋₁ 

  c = (1//3, 15//16, 8//15)

  return HM89TimeStepper( F₀, F₁, Bₘ₋₁, Bₘ, Bₘ₊₁, B₀, gₘ, gₘ₋₁ , c)
end

function stepforward!(sol, clock, ts::HM89TimeStepper, equation, vars, params, grid)
  HM89substeps!( sol, clock, ts, equation, vars, params, grid)

  clock.t += clock.dt
  
  clock.step += 1
  
  return nothing
end

function HM89substeps!(sol, clock, ts, equation, vars, params, grid)
  # we solve the equation using the most simplist 2nd order implicit methed: trapezoidal rule method
  # Consider the B_{n+1} = B_n + (Δt/2)*(f(y_{n+1/2})) = f(y), n is the time step n
  # using secant method, we define g(B) = B - B₀ - (Δt/2)*f(B'), B' = B^{n+1/2} 
  #  So that,  B_{m + 1} = B_m - g(B_m)*(B_m - B_{m-1})/(g(B_m) - g(B_{m-1})), be aware m is the iteration number m
  # If the B_{m+1} is convergence, B_{m+1} = y_{m}
  # We pick B_{m=0} = Euler(B₀), and B_{m=1} = RK3(B) as our first guess
#-------------------------------------------------------------------------------------------#
  # Define the function and var that will be used
  square_mean(A,B,C) =  mapreduce((x,y,z)->√(x*x+y*y+z*z),max,A,B,C)

  t, Δt, c  = clock.t, clock.dt, ts.c
  
  gₘ₋₁, gₘ = ts.gₘ₋₁, ts.gₘ

  B₀, Bₘ, Bₘ₋₁, Bₘ₊₁ = ts.B₀, ts.Bₘ, ts.Bₘ₋₁, ts.Bₘ₊₁

  ΔBh, ∇XJXB =  ts.F₀, ts.F₁

  ΔBx, ΔBy, ΔBz = vars.bx, vars.by, vars.bz
  
  # copy B⁰ from sol and get guess of B(m=1) from LSRK3 Method
  copyto!(B₀, sol)
  LSRK3substeps!(sol, clock, ts, equation, vars, params, grid)
  DivFreeCorrection!(sol, vars, params, grid)
  copyto!(Bₘ,  sol)
  
  # define the "pointer" and set up the guess of B(m=0) from Euler Method
  Bₕ = sol
  equation.calcN!(∇XJXB, Bₕ, t, clock, vars, params, grid)
  @. Bₘ₋₁ = B₀ + Δt*∇XJXB
  @. Bₕ = (B₀ + Bₘ₋₁)/2
  equation.calcN!(∇XJXB, Bₕ, t, clock, vars, params, grid)
  @. gₘ₋₁ = Bₘ₋₁ - B₀ - Δt*∇XJXB

  # delias the result before the iteration
  dealias!(Bₘ₋₁, grid)
  dealias!(gₘ₋₁, grid)
  dealias!(Bₘ, grid)

  ε   = 1.0;
  err = 5e-4;

  while ε > err 
    
    # get the ∇×(J × B) term from B^{n+1/2}
    @. Bₕ = (B₀ + Bₘ)/2 
    equation.calcN!(∇XJXB, Bₕ, t, clock, vars, params, grid) 
  
    @.   gₘ = Bₘ - B₀ - Δt*∇XJXB
    dealias!(gₘ, grid)
    # get the term B\^ m + 1 and de-alias the result to avoid aliasing error
    @. Bₘ₊₁ = Bₘ - gₘ*(Bₘ - Bₘ₋₁)/(gₘ - gₘ₋₁) 
    dealias!(Bₘ₊₁, grid)

    # compute the error
    @. ΔBh = (Bₘ₊₁ - Bₘ)
    dealias!(ΔBh, grid)
    ldiv!( ΔBx, grid.rfftplan, deepcopy( @view ΔBh[:,:,:,1] ) )
    ldiv!( ΔBy, grid.rfftplan, deepcopy( @view ΔBh[:,:,:,2] ) )
    ldiv!( ΔBz, grid.rfftplan, deepcopy( @view ΔBh[:,:,:,3] ) ) 
    ε = square_mean(ΔBx, ΔBy, ΔBz)

    # swap the vars for the next iteration
    copyto!(Bₘ₋₁, Bₘ)
    copyto!(Bₘ, Bₘ₊₁)
    copyto!(gₘ₋₁, gₘ)
    #@show ε
  end

  if isinf(ε) || isnan(ε); error("No Convergence!"); end

  #Compute the diffusion term and forcing using the explicit method 
  copyto!(sol, Bₘ)
  RK3linearterm!(sol, ts, clock, vars, params, grid)
  DivFreeCorrection!(sol, vars, params, grid)

  #copy the ans back to real vars
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 

  return nothing

end

function DivFreeCorrection!(sol, vars, params, grid)
#= 
   Possion Solver for periodic boundary condition
   As in VP method, ∇ ⋅ B = 0 doesn't hold, B_{t+1} = ∇×Ψ + ∇Φ -> ∇ ⋅ B = ∇² Φ
   We need to find Φ and remove it using a Poission Solver 
   Here we are using the Fourier Method to find the Φ
   In Real Space,  
   ∇² Φ = ∇ ⋅ B   
   In k-Space,  
   ∑ᵢ -(kᵢ)² Φₖ = i∑ᵢ kᵢ(Bₖ)ᵢ
   Φ = F{ i∑ᵢ kᵢ (Bₖ)ᵢ / ∑ᵢ (k²)ᵢ}
=#  

  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
  k⁻² = grid.invKrsq;
  @. vars.nonlin1  *= 0;
  @. vars.nonlinh1 *= 0;       
  ∑ᵢkᵢBᵢh_k² = vars.nonlinh1;
  ∑ᵢkᵢBᵢ_k²  = vars.nonlin1;

  # it is N not sol
  @views bxh = sol[:, :, :, params.bx_ind];
  @views byh = sol[:, :, :, params.by_ind];
  @views bzh = sol[:, :, :, params.bz_ind];

  @. ∑ᵢkᵢBᵢh_k² = -im*(kᵢ*bxh + kⱼ*byh + kₖ*bzh);
  @. ∑ᵢkᵢBᵢh_k² = ∑ᵢkᵢBᵢh_k²*k⁻²;  # Φₖ
 
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  @. bxh  -= im*kᵢ.*∑ᵢkᵢBᵢh_k²;
  @. byh  -= im*kⱼ.*∑ᵢkᵢBᵢh_k²;
  @. bzh  -= im*kₖ.*∑ᵢkᵢBᵢh_k²;

  return nothing
end

function LSRK3substeps!(sol, clock, ts, equation, vars, params, grid)
  # Low stoage 3 step RK3 method (LSRK3)
  # F0 = dt F(0)
  # p1 = p0 + c1 F0
  # F1  = dt*F(1) - F0*5/9
  # p2 = p0 + c2 F1
  # F2 = -153/128*F(1) + dt*F(2)
  # p3 = p2 + c3*F2

  t  = clock.t
  dt = clock.dt
  c  = ts.c

  equation.calcN!(ts.F₀, sol, t + dt, clock, vars, params, grid)
  @. ts.F₀ *=  dt
  @.  sol  += ts.F₀*c[1]*dt

  equation.calcN!(ts.F₁, sol, t + dt, clock, vars, params, grid)
  @. ts.F₁ *=  dt
  @. ts.F₁ -=  5/9*ts.F₀
  @.  sol  +=  c[2]*ts.F₁

  # reuse F2 = F0
  equation.calcN!(ts.F₀, sol, t + dt, clock, vars, params, grid)
  @. ts.F₀ *= dt
  @. ts.F₀ -= 153/128*ts.F₁
  @.   sol += c[3]*ts.F₀
  return nothing
end

function RK3linearterm!(sol, ts, clock, vars, params, grid)
  # LSKR3 for diffusion term 
  # F0 = dt F(0)
  # p1 = p0 + c1 F0
  # F1  = dt*F(1) - F0*5/9
  # p2 = p0 + c2 F1
  # F2 = -153/128*F(1) + dt*F(2)
  # p3 = p2 + c3*F2

  t  = clock.t
  dt = clock.dt
  c  = ts.c
  k² = grid.Krsq
  η  = params.η
  
  params.calcF!(ts.F₀, sol, t + dt, clock, vars, params, grid)
  @. ts.F₀ -=  η*k²*sol
  @. ts.F₀ *=  dt
  @.  sol  += ts.F₀*c[1]*dt

  params.calcF!(ts.F₁, sol, t + dt, clock, vars, params, grid)
  @. ts.F₁ -=  η*k²*sol
  @. ts.F₁ *=  dt
  @. ts.F₁ -=  5/9*ts.F₀
  @.  sol  +=  c[2]*ts.F₁

  # reuse F2 = F0
  params.calcF!(ts.F₀, sol, t + dt, clock, vars, params, grid)
  @. ts.F₀ -= η*k²*sol
  @. ts.F₀ *= dt
  @. ts.F₀ -= 153/128*ts.F₁
  @.   sol += c[3]*ts.F₀

  return nothing
end


