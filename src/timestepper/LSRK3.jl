struct LSRK3TimeStepper{T,TL} <: FourierFlows.AbstractTimeStepper{T}
  F₀  :: T
  F₁  :: T
   c  :: TL
end

function LSRK3TimeStepper(equation, dev::Device=CPU())
  @devzeros typeof(dev) equation.T equation.dims   F₀  F₁  

  c = (1//3, 15//16, 8//15)

  return LSRK3TimeStepper( F₀, F₁, c)
end


function stepforward!(sol, clock, timestepper, eqn, vars, params, grid)
  FourierFlows.stepforward!(sol, clock, timestepper, eqn, vars, params, grid)
end

function stepforward!(sol, clock, ts::LSRK3TimeStepper, equation, vars, params, grid)
  LSRK3substeps!(sol, clock, ts, equation, vars, params, grid)

  clock.t += clock.dt
  
  clock.step += 1
  
  return nothing
end


function LSRK3substeps!(sol, clock, ts, equation, vars, params, grid)
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

