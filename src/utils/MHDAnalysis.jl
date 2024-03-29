# ----------
# MHD Analysis Method, providing MHD related quantities function 
# ----------

"""
    ScaleDecomposition(B1,B2,B3)

Funtion of decomposing the fluctuation into different scale using the fourier method
The function will return the array containing fluctuation scale between kf[1] and kf[2]
Warning : For Periodic Maps Only
  Keyword arguments
=================
- `B1/B2/B3`: 3D physical quantites array 
- `kf` : Scale of sparation from kf[1] to kf[2] (T type: Array)
- `Lx` : Maximum Length Scale for the box(T type: Int)
"""
function ScaleDecomposition(B1::Array;kf=[1,5],Lx = 2π,T=Float32)
  nx,ny,nz = size(B1);
  grid = GetSimpleThreeDGrid(nx, Lx, T = T);
  cB1  = ScaleDecomposition(B1,grid;kf=kf)
  return cB1;
end

function ScaleDecomposition(B1::Array,grid;kf=[1,5])
  k1,k2 = minimum(kf),maximum(kf);
  nx,ny,nz = size(B1);
  dev = typeof(B1) <: Array ? CPU() : GPU();
  T   = eltype(grid);
  
  #Define the Array that will be used  
  @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h
  @devzeros typeof(dev)         T  (         nx,ny,nz) cB1
  @devzeros typeof(dev)         T  (div(nx,2)+1,ny,nz)  K  kr
  mul!(B1h, grid.rfftplan, B1);
    
  kx,ky,kz = grid.kr,grid.l,grid.m;
  @. kr = sqrt.(kx.^2 .+ ky.^2 .+ kz.^2);

  K[k2 .>= kr .>= k1] .= 1;
  @. B1h = B1h*K;
    
  ldiv!(cB1, grid.rfftplan,B1h);  
  
  return cB1;
end

function ScaleDecomposition(B1::Array,B2::Array,B3::Array;kf=[1,5],Lx = 2π,T=Float32)
  nx,ny,nz = size(B1);
  grid = GetSimpleThreeDGrid(nx, Lx, T = T);

  cB1,cB2,cB3 = ScaleDecomposition(B1,B2,B3,grid;kf=kf)
  return cB1,cB2,cB3;
end

function ScaleDecomposition(B1,B2,B3,grid;kf=[1,5])
  k1,k2 = minimum(kf),maximum(kf);
  nx,ny,nz = size(B1);
  dev = typeof(B1) <: Array ? CPU() : GPU();
  T   = eltype(grid);
    
  @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h B2h B3h 
  @devzeros typeof(dev)         T  (         nx,ny,nz) cB1 cB2 cB3
  @devzeros typeof(dev)         T  (div(nx,2)+1,ny,nz)  K   kr

  mul!(B1h, grid.rfftplan, B1); 
  mul!(B2h, grid.rfftplan, B2); 
  mul!(B3h, grid.rfftplan, B3); 
    
  kx,ky,kz = grid.kr,grid.l,grid.m;
  @. kr = sqrt.(kx.^2 .+ ky.^2 .+ kz.^2);
  K[k2 .>= kr .>= k1] .= 1;
    
  @. B1h = B1h*K;
  @. B2h = B2h*K;
  @. B3h = B3h*K;

  ldiv!(cB1, grid.rfftplan,B1h);
  ldiv!(cB2, grid.rfftplan,B2h);  
  ldiv!(cB3, grid.rfftplan,B3h);
  return cB1,cB2,cB3;  
end

"""
    h_k(iv,jv,kv)

Funtion of computing kenitic helicity hₖ using the fourier method
Warning : For Periodic Maps Only
  Keyword arguments
=================
- `iv/jv/kv`: 3D i/j/k velocity field array 
- `Lx` : Maximum Length Scale for the box(T type: Int)
"""
function h_k(iv::Array{T,3},jv::Array{T,3},kv::Array{T,3};L=2π) where T
  # V ⋅ ( ∇ × V )
  dlx,dly,dlz  = (L/size(iv)[1]),(L/size(iv)[2]),(L/size(iv)[3]);
  dV = dlx*dly*dlz;
  cV1,cV2,cV3 = Curl(iv,jv,kv;Lx=L);
  h_k = @. (cV1::Array{T,3}*iv + cV2::Array{T,3}*jv + cV3::Array{T,3}*kv)*dV
  return h_k  
end

"""
    h_m(ib,jb,kb)

Funtion of computing magnetic helicity hₘ using the fourier method
Warning : For Periodic Maps Only and we are assuming the Coulomb gauge ∇ ⋅ A = 0
  Keyword arguments
=================
- `ib/jb/kb`: 3D i/j/k magnetic field array 
- `Lx` : Maximum Length Scale for the box(T type: Int)
"""
function h_m(ib::Array{T,3},jb::Array{T,3},kb::Array{T,3}) where T
  # A ⋅ B 
  Ax,Ay,Az = VectorPotential(ib,jb,kb);
  return Ax::Array{T,3}.*ib .+ Ay::Array{T,3}.*jb .+ Az::Array{T,3}.*kb;
end

"""
    VectorPotential(B1,B2,B3)

Funtion of computing B = ∇ × A using the fourier method
Warning : We are assuming the Coulomb gauge ∇ ⋅ A = 0
  Keyword arguments
=================
- `B1/B2/B3`: 3D i/j/k magnetic field array 
- `Lx` : Maximum Length Scale for the box(T type: Int)
"""
function VectorPotential(B1::Array{T,3},B2::Array{T,3},B3::Array{T,3};L=2π) where T
  # Wrapper of actual Vector Potential function
  nx,ny,nz = size(B1);
  grid = ThreeDGrid(; nx=nx, Lx=L, T = T);
  A1,A2,A3 = VectorPotential(B1,B2,B3,grid);
  return A1,A2,A3;
end

function VectorPotential(B1,B2,B3,grid)
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
  dev = typeof(B1) <: Array ? CPU() : GPU();
  T   = eltype(grid);

  @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h B2h B3h Axh Ayh Azh
  @devzeros typeof(dev)         T  (         nx,ny,nz)  A1  A2  A3
  
  mul!(B1h, grid.rfftplan, B1); 
  mul!(B2h, grid.rfftplan, B2); 
  mul!(B3h, grid.rfftplan, B3);

  k⁻² = grid.invKrsq;
  kx,ky,kz = grid.kr,grid.l,grid.m; 

  #Actual Computation
  @. Axh = im*(ky*B3h - kz*B2h)*k⁻²;
  @. Ayh = im*(kz*B1h - kx*B3h)*k⁻²;
  @. Azh = im*(kx*B2h - ky*B1h)*k⁻²;
  
  dealias!(Axh, grid)
  dealias!(Ayh, grid)
  dealias!(Azh, grid)
  
  ldiv!(A1, grid.rfftplan, deepcopy(Axh));  
  ldiv!(A2, grid.rfftplan, deepcopy(Ayh));
  ldiv!(A3, grid.rfftplan, deepcopy(Azh));
  return A1,A2,A3;
end

"""
    getL(iv,jv,kv)

Function of computing anagular momentum L for cylindrical coordinates.
Defined direction : x/y radial dimension, z vertical dimension
Warning : using center point as a reference (r = 0 at the center)
  Keyword arguments
=================
- `iv/jv/kv`: 3D i/j/k velocity array 
- `Lx` : Maximum Length Scale for the box(T type: Int)
"""
function getL(iv::Array{T,3},jv::Array{T,3},kv::Array{T,3};L=2π) where T
  nx,ny,nz = size(iv);
  grid = GetSimpleThreeDGrid(nx, L, T = T);
  Lᵢ,Lⱼ,Lₖ  = getL(iv,jv,kv,grid)
  return Lᵢ,Lⱼ,Lₖ    
end 


function getL(iv,jv,kv,grid)
  # L = r × p => (rᵢ,rⱼ,0) × (iv,jv,kv)
  # |  i  j  k |
  # | rᵢ rⱼ  0 | = (y*kv) i - (x*kv) j + (x*jv - y*iv) k;  
  # | iv jv kv |
  x,y,z = grid.x,grid.y,grid.z;
  Li = @.  (y*kv - 0   );
  Lj = @. -( 0   - z*iv);
  Lk = @.  (x*jv - y*iv);
  return Li,Lj,Lk    
end 


"""
    spectralline(A)

Function of computing 2D/3D Spectra
Warning : For Periodic Maps Only
  Keyword arguments
=================
- `A`: 2D/3D Array 
- `Lx` : Maximum Length Scale for the box(T type: Int)
"""
function spectralline(A::Array{T,2};Lx=2π) where T
  nx,ny = size(A);
  Ak = zeros(Complex{T},div(nx,2)+1,ny);
  grid = TwoDGrid(CPU();nx,Lx,T=T);
  mul!(Ak,grid.rfftplan,A);
  kk    = @. √(grid.Krsq);
  krmax = round(Int,maximum(kk)+1);
  Pk = zeros(T,(krmax));
  kr = zeros(T,(krmax));
  for j = 1:ny::Int
    @simd for i = 1:div(nx,2)+1::Int
           r = round(Int,kk[i,j])+1;
      Pk[r] += abs(Ak[i,j]^2);
      kr[r]  = r;
    end
  end
  return Pk,kr
end

function spectralline(A::Array{T,3};Lx=2π) where T
  nx,ny,nz = size(A);
  Ak = zeros(Complex{T},div(nx,2)+1,ny,nz);
  grid = GetSimpleThreeDGrid(nx,Lx,ny,Lx,nz,Lx;T=T);
  k²,rfftplan = grid.Krsq,grid.rfftplan;
  mul!(Ak,rfftplan,A);
  kk    = @. √(k²::Array{T,3});
  krmax = round(Int,maximum(kk)+1);
  Pk = zeros(T,(krmax));
  kr = zeros(T,(krmax));
  for k = 1:nz::Int, j = 1:ny::Int
    @simd for i = 1:div(nx,2)+1::Int
           r = round(Int,kk[i,j,k])+1;
      Pk[r] += abs(Ak[i,j,k]^2);
      kr[r]  = r;
    end
  end
  return Pk,kr
end

#-------------------------------- mode analysis --------------------------------------#

"""
    SlowMode(U1,U2,U3,B1,B2,B3)

function of decomposing the slow mode
  Keyword arguments
=================
- `B1/B2/B3` : B-field in real space
- `U1/U1/U1` : target vecotr-field in to be decomposed (V/B)
- `Lx/Ly/Lz` : Box Scale
"""
function SlowMode(U1::Array,U2::Array,U3::Array, B1::Array,B2::Array,B3::Array;
              Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    
    # Wrapper for SlowMode Function
    #compute the mean field direction
    mb1,mb2,mb3 = mean(B1), mean(B2), mean(B3)
    mbb = sqrt(mb1^2 + mb2^2 + mb3^2)
    mb1,mb2,mb3 = mb1/mbb,mb2/mbb,mb3/mbb
    
    nx,ny,nz = size(B1);
    grid = MHDFlows.GetSimpleThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = eltype(U1));
    B1s,B2s,B3s = SlowMode(U1,U2,U3, (mb1,mb2,mb3), grid)
    return B1s,B2s,B3s;
end

"""
    AlfvenMode(U1,U2,U3,B1,B2,B3)

function of decomposing the Alfvenic mode
  Keyword arguments
=================
- `B1/B2/B3` : B-field in real space
- `U1/U1/U1` : target vecotr-field in to be decomposed (V/B)
- `Lx/Ly/Lz` : Box Scale
"""
function AlfvenMode(U1::Array,U2::Array,U3::Array, B1::Array,B2::Array,B3::Array;
                   Lx = 2π, Ly = Lx, Lz = Lx,T = Float32)
    
    # Wrapper for AlfMode Function
    # computing the mean field direction
    mb1,mb2,mb3 = mean(B1), mean(B2), mean(B3)
    mbb = sqrt(mb1^2 + mb2^2 + mb3^2)
    mb1,mb2,mb3 = mb1/mbb,mb2/mbb,mb3/mbb
   
    nx,ny,nz = size(B1);
    grid = MHDFlows.GetSimpleThreeDGrid(nx, Lx, ny, Ly, nz, Lz, T = T);
    B1s,B2s,B3s = AlfvenMode(U1,U2,U3,(mb1,mb2,mb3),grid)
    return B1s,B2s,B3s;
end

function SlowMode(B1,B2,B3,mb,grid)
  #funtion of computing Slow mode of B-field B_s using the fourier method
  # where B_s = \vec{B} ⋅ \vec(k × (k×B₀))
  nx,ny,nz = size(B1)
  dev = typeof(B1) <: Array ? CPU() : GPU()
  T   = eltype(grid)

  @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h B2h B3h B1sh B2sh B3sh
  @devzeros typeof(dev)         T  (         nx,ny,nz) B1s B2s B3s

  k⁻² = grid.invKrsq
  mb1,mb2,mb3 = mb  
  mul!(B1h, grid.rfftplan, B1) 
  mul!(B2h, grid.rfftplan, B2) 
  mul!(B3h, grid.rfftplan, B3)
  
  kx,ky,kz = grid.kr,grid.l,grid.m 
  # first k×B₀ vector
  vB1h = @. (ky*mb3 - kz*mb2)
  vB2h = @. (kz*mb1 - kx*mb3)
  vB3h = @. (kx*mb2 - ky*mb1)

  # second k×(k×B₀) vector
  vvB1h = @. (ky.*vB3h .- kz.*vB2h)*k⁻²
  vvB2h = @. (kz.*vB1h .- kx.*vB3h)*k⁻²
  vvB3h = @. (kx.*vB2h .- ky.*vB1h)*k⁻²
 
  # work out the \vec{B_s} = \vec{B} ⋅ \vec(k × (k×B₀))
  @. B1sh = ( B1h*vvB1h + B2h*vvB2h + B3h*vvB3h )*vvB1h
  @. B2sh = ( B1h*vvB1h + B2h*vvB2h + B3h*vvB3h )*vvB2h
  @. B3sh = ( B1h*vvB1h + B2h*vvB2h + B3h*vvB3h )*vvB3h

  ldiv!(B1s, grid.rfftplan, B1sh)  
  ldiv!(B2s, grid.rfftplan, B2sh)
  ldiv!(B3s, grid.rfftplan, B3sh)

  return B1s,B2s,B3s
end

function AlfvenMode(B1,B2,B3,mb,grid)
  #funtion of computing ALf mode of B-field B_a using the fourier method
  # where B_a = \vec{B} ⋅ \vec(k×B₀)
  nx,ny,nz = size(B1)
  dev = typeof(B1) <: Array ? CPU() : GPU()
  T   = eltype(grid)

  @devzeros typeof(dev) Complex{T} (div(nx,2)+1,ny,nz) B1h B2h B3h B1ah B2ah B3ah
  @devzeros typeof(dev)         T  (         nx,ny,nz) B1a B2a B3a

  k⁻¹ = sqrt.(grid.invKrsq)
  mb1,mb2,mb3 = mb  

  mul!(B1h, grid.rfftplan, B1) 
  mul!(B2h, grid.rfftplan, B2) 
  mul!(B3h, grid.rfftplan, B3)
  
  kx,ky,kz = grid.kr,grid.l,grid.m 
  # first k×B₀ vector
  vB1h = @. (ky*mb3 - kz*mb2)*k⁻¹
  vB2h = @. (kz*mb1 - kx*mb3)*k⁻¹
  vB3h = @. (kx*mb2 - ky*mb1)*k⁻¹

  # work out the  \vec{B_a} = \vec{B} ⋅ \vec(k×B₀)
  @. B1ah = ( B1h*vB1h + B2h*vB2h + B3h*vB3h )*vB1h
  @. B2ah = ( B1h*vB1h + B2h*vB2h + B3h*vB3h )*vB2h
  @. B3ah = ( B1h*vB1h + B2h*vB2h + B3h*vB3h )*vB3h

  ldiv!(B1a, grid.rfftplan, B1ah)  
  ldiv!(B2a, grid.rfftplan, B2ah)
  ldiv!(B3a, grid.rfftplan, B3ah)

  return B1a, B2a, B3a
end