module VectorCalculus
## A functional moduile providing the basic vector calculus operation

export 
  Curl,
  Div

using
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

function Curl(B1::Array,B2::Array,B3::Array;Lx = 2π)
    #funtion of computing ∇×Vector using the fourier method
    # fft(∇×Vector) -> im * k × V
    #| i j k  |
    #| x y z  |
    #|B1 B2 B3|
    # 
    nx,ny,nz = size(B1);
    T    = Float32;
    grid = ThreeDGrid(nx, Lx, T = T);
    
    B1h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B2h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B3h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    cBxh = copy(B1h); 
    cByh = copy(B2h);
    cBzh = copy(B3h);
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
        
    for i in 1:div(nz,2)+1, j in 1:ny, k in 1:nx
       x,y,z = grid.kr[i],grid.l[j],grid.m[k]; 
       cBxh[i,j,k] = im*(y*B3h[i,j,k] - z*B2h[i,j,k]);
       cByh[i,j,k] = im*(z*B1h[i,j,k] - x*B3h[i,j,k]);
       cBzh[i,j,k] = im*(x*B2h[i,j,k] - y*B1h[i,j,k]);
    end
    
    cB1,cB2,cB3 = zeros(T,size(B1)),zeros(T,size(B1)),zeros(T,size(B1));
    ldiv!(cB1, grid.rfftplan, deepcopy(cBxh));  
    ldiv!(cB2, grid.rfftplan, deepcopy(cByh));
    ldiv!(cB3, grid.rfftplan, deepcopy(cBzh));
    return cB1,cB2,cB3;
end


function Div(B1::Array,B2::Array,B3::Array;Lx = 2π)
    #funtion of computing ∇̇ ⋅ Vector using the fourier method
    # fft(∇̇ ⋅ Vector) -> im * k ⋅ V
    # = im* x*B1 + y*B2 + z*B3
    nx,ny,nz = size(B1);
    T    = Float32;
    grid = ThreeDGrid(nx, Lx, T = T);
    
    B1h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B2h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    B3h = zeros(ComplexF32,(div(nx,2)+1,ny,nz));
    Dot = copy(B1h);
    mul!(B1h, grid.rfftplan, B1); 
    mul!(B2h, grid.rfftplan, B2); 
    mul!(B3h, grid.rfftplan, B3);
        
    for i in 1:div(nz,2)+1, j in 1:div(ny,2), k in 1:div(nx,2)
       x,y,z = grid.kr[i],grid.l[j],grid.m[k]; 
       Dot[i,j,k] = x*B1h[i,j,k] + y*B2h[i,j,k] + z*B3h[i,j,k];
    end
    
    cB1 = zeros(T,size(B1))
    ldiv!(cB1, grid.rfftplan, deepcopy(Dot));  

    return cB1
end


end