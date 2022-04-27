module MHDAnalysis
#The module provides scale decompostion, helicity output function
export 
  ScaleDecomposition,
  h_m,h_k,
  VectorPotential

using
  Reexport

@reexport using FourierFlows

include("VectorCalculus.jl")
using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum


function ScaleDecomposition(B1::Array,B2::Array,B3::Array;kf=[1,5],Lx = 2π)
    k1,k2 = minimum(kf),maximum(kf);
    nx,ny,nz = size(B1);
    T    = Float32;
    grid = ThreeDGrid(nx, Lx, T = T);
    
    B1h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B2h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B3h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    Bxhf = copy(B1h); 
    Byhf = copy(B2h);
    Bzhf = copy(B3h);
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
    
    for i in 1:div(nz,2)+1, j in 1:ny, k in 1:nx
       x,y,z = grid.kr[i],grid.l[j],grid.m[k];
       rr    = sqrt(x^2+y^2+z^2); 
       if (( rr >=  k1) && (rr <= k2))
           Bxhf[i,j,k] = B1h[i,j,k];
           Byhf[i,j,k] = B2h[i,j,k];
           Bzhf[i,j,k] = B3h[i,j,k];
       end
    end
    cB1,cB2,cB3 = zeros(T,size(B1)),zeros(T,size(B1)),zeros(T,size(B1));
    ldiv!(cB1, grid.rfftplan,Bxhf);  
    ldiv!(cB2, grid.rfftplan,Byhf);
    ldiv!(cB3, grid.rfftplan,Bzhf);
    return cB1,cB2,cB3;
end

function h_k(iv,jv,kv;L=2π)
# V ⋅ ( ∇ × V )
	dlx,dly,dlz  = (L/size(iv)[1]),(L/size(iv)[2]),(L/size(iv)[3]);
	dV = dlx*dly*dlz;
	cV1,cV2,cV3 = Curl(iv,jv,kv;Lx=L);
	h_k = @. (cV1*iv + cV2*jv + cV3*kv)*dV
	return h_k
end

function VectorPotential(B1,B2,B3;L=2π)
#=   
    funtion of computing B = ∇ × A using the fourier method
     fft(∇×Vector) -> im * k × A
      | i j k  |
      | x y z  |  =  (y*A3 - z*A2) i - (x*A3 - z*A1) j + (x*A2 - y*A1) k
      |A1 A2 A3|
    
    Note: We are using the Coulomb gauge ∇ ⋅ A = 0
    Using the relations J = ∇ × (∇ × B) and ∇ ⋅ A = 0 in peroideric condition,
    we will arrive Jₖ = - kᵢ² Aₖ.
    
    Aₖ = (k × B)ᵢ/ kᵢ²   
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
        
    for i in 1:div(nz,2)+1, j in 1:ny, k in 1:nx
       x,y,z = grid.kr[i],grid.l[j],grid.m[k]; 
       Axh[i,j,k] = im*(y*B3h[i,j,k] - z*B2h[i,j,k])/x^2;
       Ayh[i,j,k] = im*(z*B1h[i,j,k] - x*B3h[i,j,k])/y^2;
       Azh[i,j,k] = im*(x*B2h[i,j,k] - y*B1h[i,j,k])/z^2;
       Axh[i,j,k] = ifelse(x == 0, 0,  Axh[i,j,k]);
       Ayh[i,j,k] = ifelse(y == 0, 0,  Ayh[i,j,k]);
       Azh[i,j,k] = ifelse(z == 0, 0,  Azh[i,j,k]);
    end
    
    A1,A2,A3 = zeros(T,size(B1)),zeros(T,size(B1)),zeros(T,size(B1));
    ldiv!(A1, grid.rfftplan, deepcopy(Axh));  
    ldiv!(A2, grid.rfftplan, deepcopy(Ayh));
    ldiv!(A3, grid.rfftplan, deepcopy(Azh));
    return A1,A2,A3;
    
end

function h_m(ib,jb,kb)
# A ⋅ B 
   Ax,Ay,Az = VectorPotential(ib,jb,kb);
   return Ax.*ib + Ay.*jb + Az.*kb;
end

end