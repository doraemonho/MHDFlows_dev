# ----------
# Module For Magnetic helicity correction scheme, Ref : Zenati & Vishniac 2021
# ----------

# im for dddd!

mutable struct Hm{Atrans,Aphys}
  λₙ  :: Aphys
  H₀h :: Atrans
  sk  :: Aphys
end


#--------------
function construct_function_for_struct()
  
end


# function of correction the B-field to conserve the Hm
function HmCorrection!(prob; ε = 1f-4)
  square_mean(A,B,C) = mapreduce((x,y,z)->x*x+y*y+z*z,+,A,B,C)/length(A)
  L1_err_max(dA) =  mapreduce(x->√(x*x), max, dA)

  # ---------------------------- step 0. ------------------------------------#
  # define all the variables for iteration
  ts   = prob.timestepper
  sol  = prob.sol
  grid = prob.grid
  vars = prob.vars
  params = prob.params

  @views bxh,byh,bzh = sol[:,:,params.bx_ind], sol[:,:,params.by_ind], sol[:,:,params.bz_ind]
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 
  bx , by, bz = vars.bx, vars.by, vars.bz

  k⁻² = grid.invKrsq
  k1, k2, k3 = grid.kr, grid.l, grid.m

  # define the sketch variables
  # - imag space sketch variables
  @views Axh, Ayh, Azh = sk_arr1[:,:,1], sk_arr1[:,:,2], sk_arr1[:,:,3]
  @views Jxh, Jyh, Jzh = sk_arr1[:,:,4], sk_arr1[:,:,5], sk_arr1[:,:,6]
  @views sk1, sk2, sk3 = sk_arr1[:,:,4], sk_arr1[:,:,5], sk_arr1[:,:,6] # intend to do it 
  @views Hh, ΔHh       = sk_arr2[:,:,1], sk_arr2[:,:,2]
  @views λh_next, λₙh  = sk_arr2[:,:,3], sk_arr2[:,:,4]
  @views ∂ᵢD₁ᵢⱼ∂ⱼλₙh   = sk_arr2[:,:,5], sk_arr2[:,:,6]
  @views ∇∇²∇λBh     = sk_arr1[:,:,1] # intend to do it
  @views C₁λₙh = sk_arr1[:,:,2] # intend to do it
  H₀h = params.usr_params.H₀h

  # - real space sketch variables
  @views Jx, Jy, Jz = params.usr_params.sk[:,:,:,1], params.usr_params.sk[:,:,:,2], params.usr_params.sk[:,:,:,3]
  @views Ax, Ay, Az = params.usr_params.sk[:,:,:,4], params.usr_params.sk[:,:,:,5], params.usr_params.sk[:,:,:,6]
  λₙ = params.usr_params.λₙ

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
  
  # Real or imag space? Should be real space? but Cᵢ is imag space?
  C₀ = 2 *square_mean(bx,by,bz)
  C₁  = @. bx^2 + by^2 + bz^2 + (Jx,Jy,Jz)⋅(Ax,Ay,Az) - C₀ 

  # Compute the magnetic helicity 
  H  = (Ax,Ay,Az) ⋅ (bx,by,bz)
  mul!(Hh, grid.rfftplan, H)
  @. ΔHh = Hh - H₀h # imag space

  @. λₙ  = ΔH/2/(C₀ .+ C₁)
  mul!(λₙh, grid.rfftplan, λₙ)
  
  #ComputeD₀D₁!(D₀, D₁, Axh, Ayh, Azh, grid, vars)

  # ---------------------------- step 2. ------------------------------------#
  # compute the err and iterate the λ until it is converge 
  err = 1.0

  @. λh_next = 0
  while err > ε

    # compute the C₁λₙ term
    ldiv!(λₙ, grid.rfftplan, deepcopy(λhₙ))
    C₁λₙ = @. C₁*λₙ 
    mul!(C₁λₙh, grid.rfftplan, C₁λₙ)

    # compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))
    Get_the_B∇∇²∇λB_term!(B∇∇²∇λBh, λₙ, 
                           bxh, byh, bzh, bx ,by ,bz,
                           params, vars, grid)


    # Implicit summation for computing λₙ₊₁ from eq. 25 in the paper
    for (Aᵢ,kᵢ,i) ∈ zip((Ax,Ay,Az),(kx,ky,kz), (1,2,3))
      for (Aⱼ,kⱼ,j) ∈ zip((Ax,Ay,Az),(kx,ky,kz), (1,2,3))
        
        D₁ᵢⱼ = @. δ(i,j)*(Ax,Ay,Az)⋅(Ax,Ay,Az) - Aᵢ*Aⱼ
        D₀ᵢⱼ = mean(D₁ᵢⱼ)
        D₁ᵢⱼ = @. D₁ᵢⱼ - D₀ᵢⱼ 
        Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λhₙ, D₁ᵢⱼ, 
                                 params, vars, grid) 
        @. λh_next += (@. ΔHh -2*C₁λₙh + ∂ᵢD₁ᵢⱼ∂ⱼλₙh + B∇∇²∇λBh)/(@. 2*C₀ + kᵢ*kⱼ*D₀ᵢⱼ); 
      
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
   ComputeδB!(λhₙ, Axh, Ayh, Azh, 
              bx, by, bz, bxh, byh, bzh,
              vars, params, grid)

  return nothing 

end


# compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))
function Get_the_B∇∇²∇λB_term!(B∇∇²∇λBh, λ, 
                                bx ,by ,bz,
                                params, vars, grid; ∇⁻²∇λBh = sk1)

  ∂ᵢ∇⁻²∇λB  = λbᵢ = vars.nonlin1
  ∂ᵢ∇⁻²∇λBh = Bᵢ∂ᵢ∇⁻²∇λBh = λbᵢh = vars.nonlinh1

  @. ∇⁻²∇λBh  *= 0 
  @. ∇∇²∇λBh  *= 0

  k⁻² = grid.invKrsq;

  for (kᵢ,bᵢ) ∈ zip((kx,ky,kz),(bx,by,bz))
    @. λbᵢ = λ*bᵢ
    mul!(λbᵢh, grid.rfftplan, λbᵢ)
    ∇⁻²∇λBh += im*kᵢ*λbᵢh*k⁻²
  end

  for (kᵢ,bᵢ) ∈ zip((kx,ky,kz),(bx,by,bz))
    
    @. ∂ᵢ∇⁻²∇λBh = im*kᵢ*∇⁻²∇λBh

    ldiv!(∂ᵢ∇⁻²∇λB, grid.rfftplan,∇⁻²∇λBh)

    ∂ᵢ∇⁻²∇λB = bᵢ*∂ᵢ∇⁻²∇λB

    mul!(Bᵢ∂ᵢ∇⁻²∇λBh, grid.rfftplan, ∂ᵢ∇⁻²∇λB)

    B∇∇²∇λBh += Bᵢ∂ᵢ∇⁻²∇λBh

  end

  return nothing 

end

# define the vars..
function Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λh, D₁ᵢⱼ, kᵢ, kⱼ, params, vars, grid)
  # 2 real and 2 imag sketch array
  ∂ᵢD₁ᵢⱼ∂ⱼλₙh, ∂ⱼλₙh = ...  
  ∂ⱼλₙ   = vars.nonlin1

  ∂ⱼλₙh = im*kⱼ*λh

  ldiv!(∂ⱼλₙ, grid.rfftplan, ∂ⱼλₙh)

  D₁ᵢⱼ∂ⱼλₙ = @. D₁ᵢⱼ*∂ⱼλₙ

  mul!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, grid.rfftplan, D₁ᵢⱼ∂ⱼλₙ)

  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙh = im*kᵢ*∂ᵢD₁ᵢⱼ∂ⱼλₙh

  return nothing 
end

# Compute the δA and add it back to B
function ComputeδB!(λh, Axh, Ayh, Azh, 
                    bx, by, bz, bxh, byh, bzh,
                    vars, params, grid; 
                    λBxh = sk1, λByh = sk2, λBzh = sk3)

  λbᵢ = vars.nonhin1

  @. λbᵢ = λ*bx
  mul!(λBxh, grid.rfftplan, λbᵢ)
  @. λbᵢ = λ*by
  mul!(λByh, grid.rfftplan, λbᵢ)
  @. λbᵢ = λ*bz
  mul!(λBzh, grid.rfftplan, λbᵢ)    

  ∇⁻²∇λBh = -im*(kx*λ*Bxh + ky*λ*Byh + kz*λ*Bzh)*k⁻²

  # δA = ∇×(λA) + λB - ∇∇⁻²∇⋅λB
  @. δAxh = im*(@. ky*Azh - kz*Ayh) + λBxh - im*kx*∇⁻²∇λBh
  @. δAyh = im*(@. kz*Axh - kx*Azh) + λByh - im*ky*∇⁻²∇λBh
  @. δAzh = im*(@. kx*Ayh - ky*Axh) + λBzh - im*kz*∇⁻²∇λBh

  # B = B + ∇×(δA)
  @. bxh += im*(ky*δAzh - kz*δAyh)
  @. byh += im*(kz*δAxh - kx*δAzh)
  @. bzh += im*(kx*δAyh - ky*δAxh)

  return nothing
  
end