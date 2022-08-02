# ----------
# Vector Calculus Module, Only work on peroideric boundary!
# ----------

function Curl(B1::Array,B2::Array,B3::Array;
              Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    nx,ny,nz = size(B1);
    grid = ThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
    cB1,cB2,cB3 = Curl(B1,B2,B3,grid;Lx = Lx, Ly = Ly, Lz = Lz,T = T)
    return cB1,cB2,cB3;
end

function Curl(B1::Array,B2::Array,B3::Array,grid;
              Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    #funtion of computing ∇×Vector using the fourier method
    # fft(∇×Vector) -> im * k × V
    #| i j k  |
    #| x y z  |
    #|B1 B2 B3|
    #
    nx,ny,nz = size(B1);
    B1h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    B2h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    B3h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    CB1h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    CB2h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    CB3h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
    Bᵢtmp = zeros(ComplexF32,3);  
    
    kx,ky,kz = grid.kr,grid.l,grid.m; 
    @. CB1h = im*(ky*B3h - kz*B2h);
    @. CB2h = im*(kz*B1h - kx*B3h);
    @. CB3h = im*(kx*B2h - ky*B1h);
    
    cB1,cB2,cB3 = zeros(T,size(B1)),zeros(T,size(B1)),zeros(T,size(B1));
    ldiv!(cB1, grid.rfftplan, CB1h);  
    ldiv!(cB2, grid.rfftplan, CB2h);
    ldiv!(cB3, grid.rfftplan, CB3h);
    return cB1,cB2,cB3;
end

function Div(B1::Array,B2::Array,B3::Array;
             Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    nx,ny,nz = size(B1);
    grid = ThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
    cB1 = Div(B1,B2,B3,grid;Lx = Lx, Ly = Ly, Lz = Lz,T = T);

    return cB1
end


function Div(B1::Array,B2::Array,B3::Array,grid;
             Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    #funtion of computing ∇̇ ⋅ Vector using the fourier method
    # fft(∇·Vector) -> im * k ⋅ V
    # = im* (x*B1 + y*B2 + z*B3)
    nx,ny,nz = size(B1);
    B1h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    B2h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    B3h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    Dot = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
        
    kx,ky,kz = grid.kr,grid.l,grid.m; 
    @. Dot = im*(kx*B1h+ky*B2h+kz*B3h)
    
    cB1 = zeros(T,size(B1))
    ldiv!(cB1, grid.rfftplan, deepcopy(Dot));  

    return cB1
end

function Divk(B3::Array,grid; T = Float32)
    # funtion of computing zdirection of ∇̇ ⋅ Vector using the fourier method
    # fft(∇·Vector) -> im * k ⋅ V
    # ∂(B)/∂xₖ = im*z*B3
    nx,ny,nz = size(B1);
    B3h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    Dot = copy(B3h);
    mul!(B3h, grid.rfftplan, B3);
        
    for k in 1:nz, j in 1:ny,i in 1:div(nx,2)+1 
       x,y,z = grid.kr[i],grid.l[j],grid.m[k]; 
       Dot[i,j,k] = im*z*B3h[i,j,k];
    end
    
    cB3 = zeros(T,size(B3))
    ldiv!(cB3, grid.rfftplan, deepcopy(Dot));  

    return cB3
end

function LaplaceSolver(B::Array; Lx=2π, Ly = Lx, Lz = Lx, T = Float32)
    nx,ny,nz = size(B);
    grid = ThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
    Φ   = LaplaceSolver(B,grid; Lx=2π, Ly = Lx, Lz = Lz, T = Float32);
    return Φ
end

function LaplaceSolver(B::Array,grid; Lx=2π, Ly = Lx, Lz = Lz, T = Float32)
    #=
    funtion of computing ΔΦ = B using the fourier method, must be peroidic condition
    Considering in k-space, k² Φ' = B', we would get Φ = F(B'/k²)
    =#
    nx,ny,nz = size(B);
    Φ    = zeros(T,nx,ny,nz);
    Bh   = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    mul!(Bh, grid.rfftplan, B); 
    for k in 1:nz, j in 1:ny, i in 1:div(nx,2)+1
       x,y,z = grid.kr[i],grid.l[j],grid.m[k]; 
       k² = x^2 + y^2 + z^2;
       Bh[i,j,k] = Bh[i,j,k]/k²;
       if k² == 0; Bh[i,j,k] = 0; end 
    end
    ldiv!(Φ, grid.rfftplan, deepcopy(Bh));
    return Φ;
end

function Crossproduct(A1,A2,A3,B1,B2,B3)
    C1 = @.  (A2*B3 - A3*B2);
    C2 = @. -(A1*B3 - A3*B1); 
    C3 = @.  (A1*B2 - A2*B1);
    return C1,C2,C3
end

function Dotproduct(A1,A2,A3,B1,B2,B3)
    return A1.*B1 + A2.*B2 + A3.*B3 
end

function DivVCorrection!(ux,uy,uz,grid)
#= 
   Possion Solver for periodic boundary condition
   As in VP method, ∇ ⋅ V = 0 may not hold, V = ∇×Ψ + ∇Φ -> ∇ ⋅ V = ∇² Φ
   We need to find Φ and remove it using a Poission Solver 
   Here we are using the Fourier Method to find the Φ
   In Real Space,  
   ∇² Φ = ∇ ⋅ V   
   In k-Space,  
   ∑ᵢ -(kᵢ)² Φₖ = i∑ᵢ kᵢ(Vₖ)ᵢ
   Φₖ = i∑ᵢ kᵢ(Vₖ)ᵢ/k²
   Vⱼ_new = Vₖⱼ + kⱼ i∑ᵢ kᵢ(Vₖ)ᵢ/k²; 
=#  

  T = eltype(grid);  
  nx,ny,nz = grid.nx,grid.ny,grid.nz;  
  uxh = zeros(Complex{T},(div(nx,2)+1,ny,nz));
  uyh = zeros(Complex{T},(div(nx,2)+1,ny,nz));
  uzh = zeros(Complex{T},(div(nx,2)+1,ny,nz));
  mul!(uxh, grid.rfftplan, ux); 
  mul!(uyh, grid.rfftplan, uy); 
  mul!(uzh, grid.rfftplan, uz);

  #find Φₖ
  kᵢ,kⱼ,kₖ = grid.kr,grid.l,grid.m;
  k⁻² = grid.invKrsq;
  ∑ᵢkᵢUᵢh_k² = 0 .*copy(uxh);
  ∑ᵢkᵢUᵢ_k²  = 0 .*copy(ux);  
    
  ∑ᵢkᵢUᵢh_k² = @. im*(kᵢ*uxh + kⱼ*uyh + kₖ*uzh);
  ∑ᵢkᵢUᵢh_k² = @. -∑ᵢkᵢUᵢh_k²*k⁻²;  # Φₖ
  
  # B  = B* - ∇Φ = Bᵢ - kᵢΦₖ  
  uxh  .-= kᵢ.*∑ᵢkᵢUᵢh_k²;
  uyh  .-= kⱼ.*∑ᵢkᵢUᵢh_k²;
  uzh  .-= kₖ.*∑ᵢkᵢUᵢh_k²;
  
  #Update to Real Space vars
  ldiv!(ux, grid.rfftplan, deepcopy(uxh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(uy, grid.rfftplan, deepcopy(uyh));# deepcopy() since inverse real-fft destroys its input
  ldiv!(uz, grid.rfftplan, deepcopy(uzh));# deepcopy() since inverse real-fft destroys its input
  return ux,uy,uz
end
