# ----------
# Module For Magnetic helicity correction scheme, Ref : Zenati & Vishniac 2021
# ----------
# Note : Have to check if any sketch variables is overlaped
# Are A computed correctly?
mutable struct Hm_vars{Atrans,Aphys,Aphys4D}
  λₙ  :: Aphys
  H₀h :: Atrans
  sk  :: Aphys4D
end

function construct_function_for_struct(::Dev,nx::Int,ny::Int,nz::Int; T = Float32) where Dev
 
  @devzeros Dev Complex{T} ( div(nx,2) + 1 , ny, nz    ) H₀h
  @devzeros Dev         T  (            nx , ny, nz    ) λₙ
  @devzeros Dev         T  (            nx , ny, nz, 6 ) sk
  Hm = Hm_vars(λₙ, H₀h, sk)

  return Hm, HmCorrection!
end

# function of correction the B-field to conserve the Hm
# note: this has a time constraint for weak field case from eq. 31
# we have a slightly different implementation from their paper
# we start from the equation 24. 
# ΔH - 2C₁λ + ∂ᵢD₁ᵢⱼ∂ⱼλ + B⋅∇(∇⁻²∇⋅(λB)) - 2C₀λ + ∂ᵢD₀ᵢⱼ∂ⱼλ = 0
# Where in k-space, we define a function g,
# g = λ + 2C₁λ + 2C₀λ - D₀ᵢⱼkᵢkⱼλ  - (ΔH + ∂ᵢD₁ᵢⱼ∂ⱼλ + B⋅∇(∇⁻²∇⋅(λB))) (k) 
# If the equation converage, g(λₙ₊₁) = λₙ 
# In this case, we adopt the fix point method.
function HmCorrection!(prob; ε = 1f-5)
  square_mean(A,B,C) = mapreduce((x,y,z)->x*x+y*y+z*z,+,A,B,C)/length(A)
  L2_err_max(dA) =  mapreduce(x->x*x, max, dA)
  δ(a::Int,b::Int) = ( a == b ? 1 : 0 )
    
  # ---------------------------- step 0. ------------------------------------#
  # define all the variables for iteration
  ts   = prob.timestepper
  sol  = prob.sol
  grid = prob.grid
  vars = prob.vars
  params = prob.params
  sk_arr1,sk_arr2 = ts.RHS₁,ts.RHS₂
  sk_arr3,sk_arr4 = ts.RHS₃,ts.RHS₄
    
  @views bxh,byh,bzh = sol[:,:,:,params.bx_ind], sol[:,:,:,params.by_ind], sol[:,:,:,params.bz_ind]
  ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
  ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
  ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind])) 
  bx , by, bz = vars.bx, vars.by, vars.bz

  k⁻² = grid.invKrsq
  kx, ky, kz = grid.kr, grid.l, grid.m

  # define the "pointer" of sketch variables
  # - imag space sketch variables
  @views Axh, Ayh, Azh = sk_arr1[:,:,:,1], sk_arr1[:,:,:,2], sk_arr1[:,:,:,3]
  @views Jxh, Jyh, Jzh = sk_arr1[:,:,:,4], sk_arr1[:,:,:,5], sk_arr1[:,:,:,6]
  @views sk1, sk2, sk3 = sk_arr2[:,:,:,1], sk_arr2[:,:,:,2], sk_arr2[:,:,:,3] 
  @views sk4, sk5, sk6 = sk_arr2[:,:,:,4], sk_arr2[:,:,:,5], sk_arr2[:,:,:,6]   
  @views Hh, ΔHh       = sk_arr3[:,:,:,1], sk_arr3[:,:,:,2]
  @views λh_next, λₙh  = sk_arr3[:,:,:,3], sk_arr3[:,:,:,4]
  @views ∂ᵢD₁ᵢⱼ∂ⱼλₙh   = sk_arr3[:,:,:,5]
  @views kᵢkⱼD₀ᵢⱼ, I   = sk_arr4[:,:,:,1], sk_arr4[:,:,:,2]
  @views B∇∇⁻²∇λBh   = sk_arr4[:,:,:,3] 
  @views Δλh, C₁λₙh    = sk_arr4[:,:,:,4], sk_arr4[:,:,:,5] 
  H₀h = params.usr_params.H₀h

  # - real space sketch variables
  @views Jx, Jy, Jz = params.usr_params.sk[:,:,:,1], params.usr_params.sk[:,:,:,2], params.usr_params.sk[:,:,:,3]
  @views Ax, Ay, Az = params.usr_params.sk[:,:,:,4], params.usr_params.sk[:,:,:,5], params.usr_params.sk[:,:,:,6]
  Δλ = vars.nonlin1
  λₙ = params.usr_params.λₙ

  # ---------------------------- step 1. ------------------------------------#
  # some preparation work for computing ΔH, C0, C1 D, λ₀

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
  
  # C₁ = B^2 + J ⋅ A C₀
  C₀ = 2 *square_mean(bx,by,bz)  #eq. 20
  C₁  = @. bx^2 + by^2 + bz^2 + Jx*Ax + Jy*Ay + Jz*Az  # eq. 21
  # Compute the magnetic helicity 
  H  = @. Ax*bx + Ay*by + Az*bz
  mul!(Hh, grid.rfftplan, H)
  @. ΔHh = Hh - H₀h # imag space
  dealias!(ΔHh, grid)  
  ldiv!(λₙ, grid.rfftplan, deepcopy(ΔHh))
  @. λₙ  = λₙ/5;  # eq. 27
  mul!(λₙh, grid.rfftplan, λₙ)

  # ---------------------------- step 2. ------------------------------------#
  # compute the err and iterate the λ until it is converge 
  err = 1.0
  loop_i = 0
  @. λh_next = 0
  @.       I = 1
  A² = @. Ax*Ax + Ay*Ay + Az*Az
  while err > ε ||  loop_i < 50
        
    # compute the C₁λₙ term
    ldiv!(λₙ, grid.rfftplan, deepcopy(λₙh))
    C₁λₙ = @. C₁*λₙ 
    mul!(C₁λₙh, grid.rfftplan, C₁λₙ);   dealias!(C₁λₙh, grid);  

    # compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))
    Get_the_B∇∇⁻²∇λB_term!(B∇∇⁻²∇λBh, λₙ, 
                            bx ,by ,bz,
                            params, vars, grid, ∇⁻²∇λBh = sk1)
    @.   kᵢkⱼD₀ᵢⱼ = 0   
    @. ∂ᵢD₁ᵢⱼ∂ⱼλₙh = 0
    # Implicit summation for computing λₙ₊₁ from eq. 25 in the paper
    for (Aᵢ,kᵢ,i) ∈ zip((Ax,Ay,Az), (kx,ky,kz), (1,2,3))
      for (Aⱼ,kⱼ,j) ∈ zip((Ax,Ay,Az), (kx,ky,kz), (1,2,3))
                
        δᵢⱼ  = δ(i,j)
        D₁ᵢⱼ = @.  Aᵢ*Aⱼ - δᵢⱼ*A²      
        Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λₙh, D₁ᵢⱼ, 
                                 kᵢ, kⱼ, params, vars, grid)
      end
    end
    # ΔH - 2C₁λ - ∂ᵢD₁ᵢⱼ∂ⱼλ + B⋅∇(∇⁻²∇⋅(λB)) = 0
       
    #  λₙ₊₁ = g = λₙ +  ΔH - 2C₁λ - ∂ᵢD₁ᵢⱼ∂ⱼλ + B⋅∇(∇⁻²∇⋅(λB)) = 0
    # I can just take 19...   
    @. λh_next = λₙh + 3e-3*(@. ΔHh + B∇∇⁻²∇λBh - 2*C₁λₙh - ∂ᵢD₁ᵢⱼ∂ⱼλₙh)
    #@. λh_next = (ΔHh .- 2*C₁λₙh .+ ∂ᵢD₁ᵢⱼ∂ⱼλₙh .+ B∇∇⁻²∇λBh)/(2*C₀ .- kᵢkⱼD₀ᵢⱼ)
        
    # compute the rms error in real space      
    @. Δλh = λh_next - λₙh
    ldiv!(Δλ, grid.rfftplan, Δλh)
    err = sqrt(maximum(Δλ.^2))

    # data movement for next iteration
    copyto!(λₙh, λh_next)
    @. λh_next  = 0
    dealias!(λₙh, grid)
    loop_i += 1
  end

  # ---------------------------- step 3. ------------------------------------#
  # work out the δA and then compute the B' = B₀ + ∇ × δA
  ldiv!(λₙ, grid.rfftplan, deepcopy(λₙh)) 
  ComputeδB!(λₙ, Ax, Ay, Az, 
             bx, by, bz, bxh, byh, bzh,
             vars, params, grid; 
             λBxh = sk1, λByh = sk2, λBzh = sk3,
             λAxh = sk4, λAyh = sk5, λAzh = sk6,
             δAxh = Axh, δAyh = Ayh, δAzh = Azh)

  #----------------------------- step 4. ------------------------------------#
  # compute the new Helicity in K-space and move it back to H₀h for next iteration
  # Q. I skipped the calculation of ∇∇⁻²∇⋅λB, does it affect the helicity? 
  @. Axh = im*(ky*bzh - kz*byh)*k⁻²
  @. Ayh = im*(kz*bxh - kx*bzh)*k⁻²
  @. Azh = im*(kx*byh - ky*bxh)*k⁻²
  ldiv!(Ax, grid.rfftplan, deepcopy(Axh))  
  ldiv!(Ay, grid.rfftplan, deepcopy(Ayh))
  ldiv!(Az, grid.rfftplan, deepcopy(Azh))

  # Compute the magnetic helicity 
  H  = @. Ax*bx + Ay*by + Az*bz

  mul!(H₀h, grid.rfftplan, H)

  @. sk_arr1 = 0
  @. sk_arr2 = 0
  @. sk_arr3 = 0
  @. sk_arr4 = 0
  @. params.usr_params.sk = 0
  return nothing
end

# compute the  ∑_i B̂ᵢ kᵢ (k^{-2} ∑_i( kᵢ \widehat{λB} ))
function Get_the_B∇∇⁻²∇λB_term!(B∇∇⁻²∇λBh, λ, 
                                  bx ,by ,bz,
                                  params, vars, grid; ∇⁻²∇λBh = sk1)

  ∂ᵢ∇⁻²∇λB  = λbᵢ = vars.nonlin1
  ∂ᵢ∇⁻²∇λBh = Bᵢ∂ᵢ∇⁻²∇λBh = λbᵢh = vars.nonlinh1

  @. ∇⁻²∇λBh  = 0 
  @. B∇∇⁻²∇λBh = 0

  kx, ky, kz = grid.kr, grid.l, grid.m
  k⁻² = grid.invKrsq;

  for (kᵢ,bᵢ) ∈ zip((kx,ky,kz),(bx,by,bz))
    @. λbᵢ = λ*bᵢ
    mul!(λbᵢh, grid.rfftplan, λbᵢ)
    @. ∇⁻²∇λBh += -im*kᵢ*λbᵢh*k⁻² # note: -1 is coming  ∇⁻²
  end
  dealias!(∇⁻²∇λBh, grid)
  
  for (kᵢ,bᵢ) ∈ zip((kx,ky,kz),(bx,by,bz))
    
    @. ∂ᵢ∇⁻²∇λBh = im*kᵢ*∇⁻²∇λBh

    ldiv!(∂ᵢ∇⁻²∇λB, grid.rfftplan,∇⁻²∇λBh)

    @. ∂ᵢ∇⁻²∇λB = bᵢ*∂ᵢ∇⁻²∇λB

    mul!(Bᵢ∂ᵢ∇⁻²∇λBh, grid.rfftplan, ∂ᵢ∇⁻²∇λB)

    B∇∇⁻²∇λBh += Bᵢ∂ᵢ∇⁻²∇λBh

  end
  dealias!(B∇∇⁻²∇λBh, grid)
    
  return nothing 

end

# define the vars..
function Get_the_∂ᵢD₁ᵢⱼ∂ⱼλₙ_term!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, λh, D₁ᵢⱼ, kᵢ, kⱼ, params, vars, grid)
  # 2 real and 2 imag sketch array
  ∂ⱼλₙh = D₁ᵢⱼ∂ⱼλₙh = vars.nonlinh1  
  ∂ⱼλₙ  = D₁ᵢⱼ∂ⱼλₙ  = vars.nonlin1

  @. ∂ⱼλₙh = im*kⱼ*λh

  ldiv!(∂ⱼλₙ, grid.rfftplan, ∂ⱼλₙh)

  D₁ᵢⱼ∂ⱼλₙ = @. D₁ᵢⱼ*∂ⱼλₙ

  mul!(D₁ᵢⱼ∂ⱼλₙh, grid.rfftplan, D₁ᵢⱼ∂ⱼλₙ)

  @. ∂ᵢD₁ᵢⱼ∂ⱼλₙh += im*kᵢ*D₁ᵢⱼ∂ⱼλₙh
    
  dealias!(∂ᵢD₁ᵢⱼ∂ⱼλₙh, grid)
    
  return nothing 

end

# Compute the δA and add it back to B
function ComputeδB!(λ, Ax, Ay, Az,  
                    bx, by, bz, bxh, byh, bzh,
                    vars, params, grid; 
                    λBxh = sk1, λByh = sk2, λBzh = sk3,
                    λAxh = sk4, λAyh = sk5, λAzh = sk6,
                    δAxh = sk7, δAyh = sk8, δAzh = sk9)
  
  # define the sketch variables
  λbᵢ = vars.nonlin1
  δAxh, δAyh, δAzh = λBxh ,λByh ,λBzh

  kx,ky,kz = grid.kr, grid.l, grid.m
  for (bᵢ, λbᵢh, Aᵢ, λAᵢh) ∈ zip((bx, by, bz), (λBxh ,λByh ,λBzh), (Ax, Ay, Az), (λAxh ,λAyh ,λAzh))
    @. λbᵢ = λ*bᵢ
    mul!(λbᵢh, grid.rfftplan, λbᵢ)
    @. λbᵢ = λ*Aᵢ
    mul!(λAᵢh, grid.rfftplan, λbᵢ)       
  end

  # eq. 28 : δA = ∇×(λA) + λB - ∇∇⁻²∇⋅λB 
  # since ∇×(∇ϕ) = 0 in the computation of B part, we neglect the ∇∇⁻²∇⋅λB part 
  @. δAxh = im*(ky.*λAzh .- kz.*λAyh) + λBxh
  @. δAyh = im*(kz.*λAxh .- kx.*λAzh) + λByh
  @. δAzh = im*(kx.*λAyh .- ky.*λAxh) + λBzh

  # B = B + ∇×(δA)
  @. bxh += im*(ky*δAzh - kz*δAyh)
  @. byh += im*(kz*δAxh - kx*δAzh)
  @. bzh += im*(kx*δAyh - ky*δAxh)

  return nothing
  
end