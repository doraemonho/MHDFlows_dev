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
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
    Bᵢtmp = zeros(ComplexF32,3);    
    for k in 1:nz,  j in 1:ny, i in 1:div(nx,2)+1 
       x,y,z = grid.kr[i],grid.l[j],grid.m[k];
       @. Bᵢtmp = copy(B1h[i,j,k]),copy(B2h[i,j,k]),copy(B3h[i,j,k]);
       b1h,b2h,b3h =  Bᵢtmp;
       B1h[i,j,k] = im*(y*b3h - z*b2h);
       B2h[i,j,k] = im*(z*b1h - x*b3h);
       B3h[i,j,k] = im*(x*b2h - y*b1h);
    end
    
    cB1,cB2,cB3 = zeros(T,size(B1)),zeros(T,size(B1)),zeros(T,size(B1));
    ldiv!(cB1, grid.rfftplan, B1h);  
    ldiv!(cB2, grid.rfftplan, B2h);
    ldiv!(cB3, grid.rfftplan, B3h);
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
    # = im* x*B1 + y*B2 + z*B3
    nx,ny,nz = size(B1);
    B1h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    B2h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    B3h = zeros(Complex{T},(div(nx,2)+1,ny,nz));
    Dot = copy(B1h);
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
        
    for k in 1:nz, j in 1:ny,i in 1:div(nx,2)+1 
       x,y,z = grid.kr[i],grid.l[j],grid.m[k]; 
       Dot[i,j,k] = x*B1h[i,j,k] + y*B2h[i,j,k] + z*B3h[i,j,k];
    end
    
    cB1 = zeros(T,size(B1))
    ldiv!(cB1, grid.rfftplan, deepcopy(Dot));  

    return cB1
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
