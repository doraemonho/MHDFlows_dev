include("eSSPIFRK3.jl")
include("HM89.jl")
include("hyperdiffusion.jl")

function stepforward!(sol, clock, timestepper, eqn, vars, params, grid)
  FourierFlows.stepforward!(sol, clock, timestepper, eqn, vars, params, grid)
end