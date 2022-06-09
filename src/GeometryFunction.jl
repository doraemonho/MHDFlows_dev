#The module provides Geometry convertion function
function xy_to_polar(ux,uy;L=2π)
#=
  Function for converting x-y vector to r-θ vector, using linear transform
    [x']  =  [cos(θ) -rsin(θ)][r']
    [y']     [sin(θ)  rcos(θ)][θ']
    So e_r =  cosθ ̂i + sinθ ̂j
       e_θ = -sinθ ̂j + cosθ ̂j
=#    
  nx,ny,nz = size(ux);  
  dev  = CPU();
  Lx = Ly = L;
  T  = Float32;
  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)
  Ur = zeros(nx,ny,nz);
  Uθ = zeros(nx,ny,nz);
  for j ∈ 1:ny, i ∈ 1:nz
    r = sqrt(grid.x[i]^2+grid.y[j]^2);
    θ = atan(grid.y[j],grid.x[i]) ;
    θ = isnan(θ) ? π/2 : θ;
    sinθ = sin(θ);
    cosθ = cos(θ);    
    Uθ[i,j,:] .= @. -sinθ*ux[i,j,:] + cosθ*uy[i,j,:];    
    Ur[i,j,:] .= @.  cosθ*ux[i,j,:] + sinθ*uy[i,j,:];    
  end
    return Ur,Uθ;
end
