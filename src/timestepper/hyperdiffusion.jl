# ----------
# Implicit timeStepper for hyper-diffussion
# ----------

# LSRK3 constant
const  c = (1//3, 15//16, 8//15)

# Allowed Timesteppers
const allowedtimesteppers = ("RK4","LSR")

function Implicitdiffusion!(prob)
  # we solve the equation using the most simplist 2nd order implicit methed: trapezoidal rule method
  # Consider the y_{n+1} = y_n + (Δt/2)*(f(t_n, y_n) + f(t_{n+1}, y_{n+1})) = f(y)
  # using fix point method, we define g = y_{n+1} - f(y)
  # note : We arrive at y_{n + 1} = y_n + (Δt)*f(t_{n+1/2}, y_{n+1/2})) 
  # If the y_n is convergence, y_{n+1} = y_{n}

  # check if user declare the correct time intragtion solver
  if string(nameof(typeof(prob.timestepper)))[1:3] ∉ allowedtimesteppers
    error("Implicit diffusion Solver only support RK4 /LSRK4 non-linear term solver!")
  end


#-------------------------------------------------------------------------------------------#
  # Define the function and var that will be used
  square_mean(A,B,C) =  mapreduce((x,y,z)->√(x*x+y*y+z*z),max,A,B,C)

  sol, clock, ts, vars, params, grid = 
                        prob.sol, prob.clock, prob.timestepper,
                        prob.vars, prob.params, prob.grid

  t, Δt  = clock.t, clock.dt, ts
      k² = grid.Krsq
  nν,hν  = params.nν, params.hν
  
  if  string(nameof(typeof(ts)))[1:3] == "RK4"
    sol₀, sol₁, solₙ = ts.RHS₃, ts.RHS₁, ts.RHS₂
  else
    sol₀, sol₁, solₙ = sol, ts.RHS, ts.S²
  end

  Δux, Δuy, Δuz = vars.ux, vars.uy, vars.uz
  
  # copy sol₀ from sol and get guess of B\^{n+1} from LSRK3 Method
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]));
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]));
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]));
  if prob.flag.b
    nη,hη  = params.nη, params.hη
    ΔBx, ΔBy, ΔBz = vars.bx, vars.by, vars.bz
    ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]));
    ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]));
    ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])); 
  end

  RK3diffusion!(prob)
  copyto!(sol₁, sol)
  mul!( (@view sol₀[:, :, :, params.ux_ind]) , grid.rfftplan, vars.ux);
  mul!( (@view sol₀[:, :, :, params.uy_ind]) , grid.rfftplan, vars.uy);
  mul!( (@view sol₀[:, :, :, params.uz_ind]) , grid.rfftplan, vars.uz);
  if prob.flag.b
    mul!( (@view sol₀[:, :, :, params.bx_ind]) , grid.rfftplan, vars.bx);
    mul!( (@view sol₀[:, :, :, params.by_ind]) , grid.rfftplan, vars.by);
    mul!( (@view sol₀[:, :, :, params.bz_ind]) , grid.rfftplan, vars.bz); 
  end

  ε   = [1.0];
  εi  = [0.0, 0.0]
  err = 5e-3;

  while ε[1] > err 
    #y_{n + 1} = y_n + (Δt)*f(t_{n+1/2}, y_{n+1/2})) 
    @. solₙ[:,:,:,1:3] = sol₀[:,:,:,1:3] - 0.5*Δt*hν*(k²)^nν*(sol₀[:,:,:,1:3] + sol₁[:,:,:,1:3]) 

    # compute the error
    Δsol = @. solₙ[:,:,:,1:3] - sol₁[:,:,:,1:3]
    ldiv!( Δux, grid.rfftplan, deepcopy( @view Δsol[:,:,:,1] ) )
    ldiv!( Δuy, grid.rfftplan, deepcopy( @view Δsol[:,:,:,2] ) )
    ldiv!( Δuz, grid.rfftplan, deepcopy( @view Δsol[:,:,:,3] ) )

    # determine the maximum error
    ε[1] = square_mean(Δux, Δuy, Δuz)

    if prob.flag.b && nη > 0
      #y_{n + 1} = y_n + (Δt)*f(t_{n+1/2}, y_{n+1/2})) 
      @. solₙ[:,:,:,4:6] = sol₀[:,:,:,4:6] - 0.5*Δt*hη*(k²)^nη*(sol₀[:,:,:,4:6] + sol₁[:,:,:,4:6]) 

      # compute the error
      Δsol = @. solₙ[:,:,:,4:6] - sol₁[:,:,:,4:6]
      ldiv!( ΔBx, grid.rfftplan, deepcopy( @view Δsol[:,:,:,1] ) )
      ldiv!( ΔBy, grid.rfftplan, deepcopy( @view Δsol[:,:,:,2] ) )
      ldiv!( ΔBz, grid.rfftplan, deepcopy( @view Δsol[:,:,:,3] ) )
      ε[1] = maximum( (ε[1], square_mean(ΔBx, ΔBy, ΔBz)) )
    end
    # copy to Bⁿ to be B¹
    copyto!(sol₁, solₙ)
  end

  copyto!(sol, sol₁)

  #copy the ans back to real vars  
  ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]))
  ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]))
  ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind])) 
  if prob.flag.b
    ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
    ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
    ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 
  end

  return nothing

end

function RK3diffusion!(prob)
  # LSKR3 for diffusion term 
  # F0 = dt F(0)
  # p1 = p0 + c1 F0
  # F1  = dt*F(1) - F0*5/9
  # p2 = p0 + c2 F1
  # F2 = -153/128*F(1) + dt*F(2)
  # p3 = p2 + c3*F2

  sol, clock, ts, vars, params, grid = 
                        prob.sol, prob.clock, prob.timestepper,
                        prob.vars, prob.params, prob.grid


  t  = clock.t
  dt = clock.dt
  k² = grid.Krsq
  nν,nη = params.nν, params.nη 
  hν,hη = params.hν,  params.hη

  if  string(nameof(typeof(ts)))[1:3] == "RK4"
    F₀, F₁ = ts.RHS₁, ts.RHS₂
  else
    F₀, F₁ = ts.S², ts.RHS
  end

  @views vsol, vF₀, vF₁ = sol[:,:,:,1:3], F₀[:,:,:,1:3], F₁[:,:,:,1:3]
  @.  vF₀  = -hν*(k²)^nν*vsol*dt
  @. vsol += vF₀*c[1]*dt

  @.  vF₁  = -hν*(k²)^nν*vsol*dt
  @.  vF₁ -=  5/9*vF₀
  @. vsol +=  c[2]*vF₁

  # reuse F2 = F0
  @.  vF₀  = -hν*(k²)^nν*vsol*dt
  @.  vF₀ -= 153/128*vF₁
  @. vsol += c[3]*vF₀

  if prob.flag.b && nη > 0
    @views bsol, bF₀, bF₁ = sol[:,:,:,4:6], F₀[:,:,:,4:6], F₁[:,:,:,4:6]
    @.  bF₀  = -hη*(k²)^nη*bsol*dt
    @. bsol += bF₀*c[1]*dt

    @.  bF₁  = -hη*(k²)^nη*bsol*dt
    @.  bF₁ -=  5/9*bF₀
    @. bsol +=  c[2]*bF₁

    # reuse F2 = F0
    @.  bF₀  = -hη*(k²)^nη*bsol*dt
    @.  bF₀ -= 153/128*bF₁
    @. bsol += c[3]*bF₀
  end
  return nothing
end
