# ----------
# Module For Magnetic helicity correction scheme, Ref : Zenati & Vishniac 2021
# ----------


# function of correction the B-field to conserve the Hm
function HmCorrection!(prob; ε = 1f-4)
  L1_err_max(A,B) =  mapreduce(x->√(x*x - y*y),max,A,B)
  # define the variables for iteration

  # some preparation work for computing ΔH, C0, C1 D, λ₀, real of i space?
  H  = 1;
  # Real or imag space? Should be real space? but Cᵢ is imag space?
  C₀ = 2 *square_mean(bxh,byh,bzh)
  @. C₁ = bxh^2 + byh^2 + bzh^2 + (Jxh,Jyh,Jzh)⋅(Axh,Ayh,Azh) - C₀  
  @. ΔH = H - H0 
  @. λₙ = ΔH/2/(C₀ .+ C₁)
  ComputeD₀D₁!(D₀, D₁, Axh, Ayh, Azh, grid, vars)

  # compute the err 
  err = 1.0

  # iterate the λ until the err is converge 
  @. λ_next = 0
  while err > ε

    # compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))
    Get_the_∇∇²∇λB_term!(∇∇²∇λBh, λ, 
                           bxh, byh, bzh, bx ,by ,bz,
                           params, vars, grid)

    # Implicit summation for computing λₙ₊₁ from eq. 25 in the paper
    for (kᵢ,i) ∈ zip((k1,k2,k3), (1,2,3))
      for (kⱼ,j) ∈ zip((k1,k2,k3), (1,2,3))
        Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λ, @views D₁[i,j], 
                                 params, vars, grid)
        @. λ_next += (@. ΔH + -2*C₁*λₙ + ∂ᵢD₁ᵢⱼ∂ⱼλₙh + ∇∇²∇λBh)/(@. 2*C₀ + kᵢ*kⱼ*D₀[i,j]); 
      end
    end
    # compute the rms error 
    err = L1_err_max(λ_next,λₙ)

    # data movement for next iteration
    copyto!(λₙ, λ_next)

    @. λ_next  = 0

  end

  # work out the A' = A + δA and then compute the B' = ∇ × A'

  # data movement for next iteration
  copyto!(H₀,H)

  return nothing 

end

function Get_the_∇∇²∇λB_term!(∇∇²∇λBh, λ, 
                                bxh, byh, bzh, bx ,by ,bz,
                                params, vars, grid)

#define the vars...

# compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))

  @.∇⁻²∇λBh  *= 0 
  @.∇∇²∇λBh *= 0
  k⁻² = grid.invKrsq;
  for (kᵢ,bᵢh) ∈ zip((kx,ky,kz),(bxh,byh,bzh))
    ∇⁻²∇λBh += kᵢ*bᵢh*k⁻²
  end

  for (kᵢ,bᵢ) ∈ zip((kx,ky,kz),(bx,by,bz))
    ∂ᵢ∇⁻²∇λBh = ∇⁻²∇λBh*kᵢ
    ldiv!(∂ᵢ∇⁻²∇λB, grid.rfftplan,∇⁻²∇λBh)

    ∂ᵢ∇⁻²∇λB = bᵢ*∂ᵢ∇⁻²∇λB

    mul!(Bᵢ∂ᵢ∇⁻²∇λBh, grid.rfftplan, ∂ᵢ∇⁻²∇λB)

    ∇∇²∇λBh += ∂ᵢ∇⁻²∇λBh

  end

  return nothing 

end

# D should be in real space...
function ComputeD₀D₁!(D₀, D₁, Axh, Ayh, Azh, grid, vars)
  
  δ(a::Int,b::Int) = ( a == b ? 1 : 0 )
  
  D₁ᵢⱼ = vars.nonhin1

  A² = (Axh,Ayh,Azh)⋅(Axh,Ayh,Azh)
  
  for (i,Aᵢ) ∈ zip((1,2,3),(Axh,Ayh,Azh))
    for (j,Aⱼ) ∈ zip((1,2,3),(Axh,Ayh,Azh))
  
      @. D₁[i,j] = δ(i,j)*A^2 - Aᵢ*Aⱼ
      
      mul!(D₁ᵢⱼ, grid.rfftplan, @views D₀[i,j])
      
      D₀[i,j] = mean(D₁ᵢⱼ)
      
      @. D₁[i,j] -= D₀[i,j]
  
    end
  end
  
  return nothing 

end

# define the vars..
function Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λ, D₁ᵢⱼ, params, vars, grid)
  
  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙh = 0  

  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙ = 0

  ∂ⱼλₙh = kⱼ*λ

  ldiv!(∂ⱼλₙ, grid.rfftplan, ∂ⱼλₙh)

  @. D₁ᵢⱼ∂ⱼλₙh = D₁ᵢⱼ*∂ⱼλₙ

  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙ = kᵢ*D₁ᵢⱼ∂ⱼλₙh

  mul!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, grid.rfftplan, ∂ᵢD₁ᵢⱼ∂ⱼλₙ)

  return nothing 
end