# ----------
# Problem Generation Module : Shearingbox Module
# ----------
function Setup_Shearingbox!(prob; q = 0.0, Ω = 0.0, U₀x = [], U₀y = [])
  @assert prob.flag.s == true

  grid = prob.grid;
  T    = eltype(grid)
  params = prob.params;
  usr_params = params.usr_params;
  Lx,Ly = grid.Lx,grid.Ly;
  
  τΩ = abs(Lx/Ly/q/Ω)
  usr_params.τΩ = T(τΩ)
  usr_params.Ω  = T(Ω)
  usr_params.q  = T(q)
  copyto!(usr_params.ky₀ , grid.l)

  params.usr_params
  if U₀x !=[]
    @assert eltype(U₀x) == eltype(U₀y)  
    for (Uᵢ,prob_Uᵢ,prob_Uᵢh) in zip([U₀x,U₀y],
                            [usr_params.U₀x,usr_params.U₀y], [usr_params.U₀xh,usr_params.U₀yh])
  
      copyto!(prob_Uᵢ,Uᵢ);
      mul!(prob_Uᵢh , grid.rfftplan, prob_Uᵢ);
    end
  else
    U₀x,U₀y = Get_shear_profile(grid,q,Ω);
    for (Uᵢ,prob_Uᵢ,prob_Uᵢh) in zip([U₀x,U₀y],
                            [usr_params.U₀x,usr_params.U₀y], [usr_params.U₀xh,usr_params.U₀yh])

      copyto!(prob_Uᵢ,Uᵢ);
      mul!(prob_Uᵢh , grid.rfftplan, prob_Uᵢ);
    end
  end
  return nothing

end


function Get_shear_profile(grid,q::AbstractFloat,Ω::AbstractFloat)
  # U0 ≡ −qΩ \hat{y} - > - qΩ x 
  @devzeros typeof(CPU()) eltype(grid) (grid.nx, grid.ny, grid.nz) U₀x U₀y
  @. U₀x = - q*Ω;

  return U₀x,U₀y
end