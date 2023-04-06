# ----------
# Module For Magnetic helicity correction scheme, Ref : Zenati & Vishniac 2021
# ----------


# im for dddd!

# im for dddd!

# im for dddd!

Question remain : which I should declare and which I dont?


#ideal beofre equation 24, all the stuff is in real-space
# but k-spacein eq 25

mutable struct Hm{Atrans,Aphys}
  D₀ :: Aphys
  H  :: Aphys
  Hh :: Aphys
  H₀h:: Atrans
  λ  :: Aphys
  λh :: Atrans
  k₁ :: Atrans
  k₂ :: Atrans
  k₃ :: Atrans 
end


# function of correction the B-field to conserve the Hm
function HmCorrection!(prob; ε = 1f-4)
  L1_err_max(dA) =  mapreduce(x->√(x*x), max, dA)

  # ---------------------------- step 0. ------------------------------------#
  # define the variables for iteration
  sol  = prob.sol
  grid = prob.grid
  vars = prob.vars
  params = prob.params

  @views bxh,byh,bzh = sol[:,:,params.bx_ind], sol[:,:,params.by_ind], sol[:,:,params.bz_ind]
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]));
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]));
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])); 
  bx , by, bz = vars.bx, vars.by, vars.bz

  # define the sketch variables
  # - imag space sketch variables
  Jxh, Jyh, Jzh = ...
  Axh, Ayh, Azh = ...
  Hh, ΔHh       = ...
  λh_next, λₙh  = ...
  ∂ᵢD₁ᵢⱼ∂ⱼλₙh   = ...
  C₁λₙh = ...
  # - real space sketch variables
  Jx, Jy, Jz = ...
  Ax, Ay, Az = ...
  λ  = ...
  D₁ = .... # is something 9*9
  D₀ = ....
  C₁ = ....

  # ---------------------------- step 1. ------------------------------------#
  # some preparation work for computing ΔH, C0, C1 D, λ₀, real of i space?

  # Compute Vector Potential & current by assuming the Coulomb gauge ∇ ⋅ A = 0
  @. Jxh = im*(ky*bzh - kz*byh)
  @. Jyh = im*(kz*bxh - kx*bzh)
  @. Jzh = im*(kx*byh - ky*bxh)
  ldiv!(Jx, grid.rfftplan, deepcopy(Jxh))  
  ldiv!(Jy, grid.rfftplan, deepcopy(Jyh))
  ldiv!(Jz, grid.rfftplan, deepcopy(Jzh))

  @. Axh = Jxh*k⁻²
  @. Ayh = Jyh*k⁻²
  @. Azh = Jzh*k⁻²
  ldiv!(Ax, grid.rfftplan, deepcopy(Axh))  
  ldiv!(Ay, grid.rfftplan, deepcopy(Ayh))
  ldiv!(Az, grid.rfftplan, deepcopy(Azh))

  # Compute the magnetic helicity 
  Hh  = (Axh,Ayh,Azh) ⋅ (bxh,byh,bzh)

  # Real or imag space? Should be real space? but Cᵢ is imag space?
  C₀ = 2 *square_mean(bx,by,bz)
  @. C₁  = bx^2 + by^2 + bz^2 + (Jx,Jy,Jz)⋅(Ax,Ay,Az) - C₀  
  @. ΔHh = Hh - H₀h # imag space
  @. λₙh = ΔHh/2/(C₀ .+ C₁)
  ComputeD₀D₁!(D₀, D₁, Axh, Ayh, Azh, grid, vars)

  # ---------------------------- step 2. ------------------------------------#
  # compute the err and iterate the λ until the err is converge 
  err = 1.0

  @. λh_next = 0
  while err > ε

    # compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))
    Get_the_∇∇²∇λB_term!(∇∇²∇λBh, λₙ, 
                           bxh, byh, bzh, bx ,by ,bz,
                           params, vars, grid)

    # compute the C₁λₙ term
    @. C₁λₙ = C₁*λₙ 
    ldiv!(λₙ, grid.rfftplan, λhₙ)
    mul!(C₁λₙh, grid.rfftplan, C₁λₙ)

    # Implicit summation for computing λₙ₊₁ from eq. 25 in the paper
    for (kᵢ,i) ∈ zip((k1,k2,k3), (1,2,3))
      for (kⱼ,j) ∈ zip((k1,k2,k3), (1,2,3))
        Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λ, @views D₁[i,j], 
                                 params, vars, grid) 
        @. λh_next += (@. ΔHh -2*C₁λₙh + ∂ᵢD₁ᵢⱼ∂ⱼλₙh + ∇∇²∇λBh)/(@. 2*C₀ + kᵢ*kⱼ*D₀[i,j]); 
      end
    end

    # compute the rms error, may have to switch to real space
    @. Δλh = λh_next - λₙh
    ldiv!(Δλ, grid.rfftplan, Δλh)
    err = L1_err_max(Δλ)

    # data movement for next iteration
    copyto!(λhₙ, λh_next)

    @. λh_next  = 0

  end

  # ---------------------------- step 3. ------------------------------------#
  # work out the A' = A + δA and then compute the B' = ∇ × A'
  ComputeδB!()  


  # data movement for next iteration
  copyto!(H₀h,H)

  return nothing 

end


# It is wrong, we have to do imag -> real
function Get_the_∇∇²∇λB_term!(∇∇²∇λBh, λ, 
                                bx ,by ,bz,
                                params, vars, grid)

#define the vars...

# compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))

  @.∇⁻²∇λBh  *= 0 
  @.∇∇²∇λBh *= 0
  k⁻² = grid.invKrsq;
  for (kᵢ,bᵢ) ∈ zip((kx,ky,kz),(bx,by,bz))
    @. λbᵢ = λ*bᵢ
    mul!(λbᵢh, grid.rfftplan, λbᵢ)
    ∇⁻²∇λBh += im*kᵢ*λbᵢh*k⁻²
  end

  for (kᵢ,bᵢ) ∈ zip((kx,ky,kz),(bx,by,bz))
    
    ∂ᵢ∇⁻²∇λBh = im*kᵢ*∇⁻²∇λBh

    ldiv!(∂ᵢ∇⁻²∇λB, grid.rfftplan,∇⁻²∇λBh)

    ∂ᵢ∇⁻²∇λB = bᵢ*∂ᵢ∇⁻²∇λB

    mul!(Bᵢ∂ᵢ∇⁻²∇λBh, grid.rfftplan, ∂ᵢ∇⁻²∇λB)

    ∇∇²∇λBh += ∂ᵢ∇⁻²∇λBh

  end

  return nothing 

end

# D should be in real space...
function ComputeD₀D₁!(D₀, D₁, Ax, Ay, Az, grid, vars)
  
  δ(a::Int,b::Int) = ( a == b ? 1 : 0 )
  
  D₁ᵢⱼ = vars.nonhin1

  A² = (Ax,Ay,Az)⋅(Ax,Ay,Az)
  
  for (i,Aᵢ) ∈ zip((1,2,3),(Ax,Ay,Az))
    for (j,Aⱼ) ∈ zip((1,2,3),(Ax,Ay,Az))
  
      @. D₁[i,j] = δ(i,j)*A^2 - Aᵢ*Aⱼ
      
      mul!(D₁ᵢⱼ, grid.rfftplan, @views D₀[i,j])
      
      D₀[i,j] = mean(D₁ᵢⱼ)
      
      @. D₁[i,j] -= D₀[i,j]
  
    end
  end
  
  return nothing 

end

# define the vars..
function Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λh, D₁ᵢⱼ, kᵢ, kⱼ, params, vars, grid)
  
  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙh = 0  
  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙ  = 0

  ∂ⱼλₙh = im*kⱼ*λh

  ldiv!(∂ⱼλₙ, grid.rfftplan, ∂ⱼλₙh)

  @. D₁ᵢⱼ∂ⱼλₙ = D₁ᵢⱼ*∂ⱼλₙ

  mul!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, grid.rfftplan, D₁ᵢⱼ∂ⱼλₙ)

  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙh = im*kᵢ*∂ᵢD₁ᵢⱼ∂ⱼλₙh

  return nothing 
end

# It seems that we should do this in real space
function ComputeδB!(λh, Ax, Ay, Az, vars, params)
  # δA = ∇×(λA) + λB - ∇∇⁻²∇⋅λB

  # (?) Am I do this correctly?
  ∇⁻²∇λBh = -im*(kx*λ*Bxh + ky*λ*Byh + kz*λ*Bzh)*k⁻²
  
  # δA = ∇×(λA) + λB - ∇∇⁻²∇⋅λB
  @. δAxh = im*(@. ky*Azh - kz*Byh) + λ*Bxh - im*kx*∇⁻²∇λBh
  @. δAyh = im*(@. kz*Axh - kx*Azh) + λ*Byh - im*ky*∇⁻²∇λBh
  @. δAzh = im*(@. kx*Ayh - ky*Axh) + λ*Bzh - im*kz*∇⁻²∇λBh

  # B = B + ∇×(δA)
  @. Bxh += im*(ky*δAzh - kz*δAyh)
  @. Byh += im*(kz*δAxh - kx*δAzh)
  @. Bzh += im*(kx*δAyh - ky*δAxh)

  return nothing
  
end