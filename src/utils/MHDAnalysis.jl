# ----------
# MHD Analysis Method, providing MHD related quantities function 
# ----------


# Scale Decomposition FUnction
function ScaleDecomposition(B1::Array;kf=[1,5],Lx = 2π)
    nx,ny,nz = size(B1);
    T    = Float32;
    grid = ThreeDGrid(nx, Lx, T = T);

    cB1  = ScaleDecomposition(B1,grid;kf=kf,Lx = Lx)
    return cB1;
end

function ScaleDecomposition(B1::Array,grid;kf=[1,5],Lx = 2π)
    k1,k2 = minimum(kf),maximum(kf);
    nx,ny,nz = size(B1);
    T    = eltype(grid);
    
    B1h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    mul!(B1h, grid.rfftplan, B1); 
    
    kx,ky,kz = grid.kr,grid.l,grid.m;
    kr = @. sqrt(kx^2 + ky^2 + kz^2);
    K  = zeros(T,size(kr));
    K[k2 .>= kr .>= k1] .= 1;
    @. B1h = B1h*K;
    
    cB1 = zeros(T,size(B1));
    ldiv!(cB1, grid.rfftplan,B1h);  
    return cB1;
end

function ScaleDecomposition(B1::Array,B2::Array,B3::Array;kf=[1,5],Lx = 2π)
    nx,ny,nz = size(B1);
    T    = Float32;
    grid = ThreeDGrid(nx, Lx, T = T);

    cB1,cB2,cB3 = ScaleDecomposition(B1,B2,B3,grid;kf=kf,Lx = Lx)
    return cB1,cB2,cB3;
end

function ScaleDecomposition(B1::Array,B2::Array,B3::Array,grid;kf=[1,5],Lx = 2π)
    k1,k2 = minimum(kf),maximum(kf);
    nx,ny,nz = size(B1);
    T    = eltype(grid);
    
    B1h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B2h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B3h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3); 
    
    kx,ky,kz = grid.kr,grid.l,grid.m;
    kr = @. sqrt(kx^2 + ky^2 + kz^2);
    K  = zeros(T,size(kr));
    K[k2 .>= kr .>= k1] .= 1;
    
    @. B1h = B1h*K;
    @. B2h = B2h*K;
    @. B3h = B3h*K;

    cB1 = zeros(T,size(B1));
    cB2 = zeros(T,size(B2));
    cB3 = zeros(T,size(B3));
    ldiv!(cB1, grid.rfftplan,B1h);
    ldiv!(cB2, grid.rfftplan,B2h);  
    ldiv!(cB3, grid.rfftplan,B3h);
    return cB1,cB2,cB3;
end

#Kenitic Helicity
function h_k(iv,jv,kv;L=2π)
# V ⋅ ( ∇ × V )
	dlx,dly,dlz  = (L/size(iv)[1]),(L/size(iv)[2]),(L/size(iv)[3]);
	dV = dlx*dly*dlz;
	cV1,cV2,cV3 = Curl(iv,jv,kv;Lx=L);
	h_k = @. (cV1*iv + cV2*jv + cV3*kv)*dV
	return h_k
end

#Megnetic Helicity
function h_m(ib,jb,kb)
# A ⋅ B 
   Ax,Ay,Az = VectorPotential(ib,jb,kb);
   return Ax.*ib + Ay.*jb + Az.*kb;
end

#Vector Potential
function VectorPotential(B1,B2,B3;L=2π)
#=   
    funtion of computing B = ∇ × A using the fourier method
     fft(∇×Vector) -> im * k × A
      | i j k  |
      | x y z  |  =  (y*A3 - z*A2) i - (x*A3 - z*A1) j + (x*A2 - y*A1) k
      |A1 A2 A3|
    
    Note: We are using the Coulomb gauge ∇ ⋅ A = 0
    Using the relations J = ∇ × (∇ × B) and ∇ ⋅ A = 0 in peroideric condition,
    we will arrive Jₖ = - k² Aₖ.
    
    Aₖ = (k × B)ᵢ/ k²   
=#
    nx,ny,nz = size(B1);
    T    = Float32;
    grid = ThreeDGrid(nx, L, T = T);
    
    B1h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B2h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B3h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    Axh = copy(B1h); 
    Ayh = copy(B2h);
    Azh = copy(B3h);
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);

    k⁻² = grid.invKrsq;
    kx,ky,kz = grid.kr,grid.l,grid.m; 

    #Actual Computation
    @. Axh = im*(ky*B3h - kz*B2h)*k⁻²;
    @. Ayh = im*(kz*B1h - kx*B3h)*k⁻²;
    @. Azh = im*(kx*B2h - ky*B1h)*k⁻²;
    
    A1,A2,A3 = zeros(T,size(B1)),zeros(T,size(B1)),zeros(T,size(B1));
    ldiv!(A1, grid.rfftplan, deepcopy(Axh));  
    ldiv!(A2, grid.rfftplan, deepcopy(Ayh));
    ldiv!(A3, grid.rfftplan, deepcopy(Azh));
    return A1,A2,A3;
    
end

#Checking for anagular momentum using center point as a reference (r = 0 at the center)
function getL(iv,jv,kv;L=2π)
    nx,ny,nz = size(iv);
    T    = Float32;
    grid = ThreeDGrid(nx, L, T = T);
    Lᵢ,Lⱼ,Lₖ  = getL(iv,jv,kv,grid)
    return Lᵢ,Lⱼ,Lₖ    
end 

function getL(iv,jv,kv,grid)
    nx,ny,nz = size(iv);
    T    = Float32;
    Lᵢ,Lⱼ,Lₖ = zeros(nx,ny,nz),zeros(nx,ny,nz),zeros(nx,ny,nz);
    for k in 1:nz, j in 1:ny, i in 1:nx
       x,y,z = grid.x[i],grid.y[j],grid.z[k];
       r = sqrt(x^2 + y^2);
       # L = r × p => iv,jv,kv
       # |  i  j  k |
       # |  x  y  z | = (y*kv - z*jv) i - (x*kv - z*iv) j + (x*jv - y*iv) k;  
       # | iv jv kv |
       Lᵢ[i,j,k] =   (y*kv[i,j,k] - z*jv[i,j,k]);
       Lⱼ[i,j,k] =  -(x*kv[i,j,k] - z*iv[i,j,k]);
       Lₖ[i,j,k] =   (x*jv[i,j,k] - y*iv[i,j,k]);
    end    
    return Lᵢ,Lⱼ,Lₖ    
end 
