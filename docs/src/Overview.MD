# Overview

MHDFlow is the Hydrodynamic/Magnetohydrodynamic solver using psudo-spectral method. In general, MHDFlows solves the following equations :

$$
\begin{align}
\frac{\partial \vec{v} }{\partial t} + ( \vec{v} \cdot \nabla)\vec{v} = -\nabla P + \vec{J} \times \vec{B} + \nu\nabla^2\vec{v},  \\
\frac{\partial \vec{B} }{\partial t}  = \nabla\times(\vec{v}\times\vec{B}) + \eta\nabla^2 \vec{B},  \\
\nabla \cdot \vec{B} = 0, \\ 
\nabla \cdot \vec{v} = 0  
\end{align}
$$

$$ \vec{B} = 0 $$ in hydro regime. Using the divergence free condition to simplify both velocity and induction equation then convert them to spectral space by doing a fourier transform, we arrive at 

$$
\begin{align}
\frac{\partial \hat{v_\alpha} }{\partial t} = -i k_i (δ_{a,j} - k_\alpha k_j k⁻²)(\widehat{b_i b_j} - \widehat{u_i u_j}), \\
\frac{\partial \hat{B_\alpha} }{\partial t} = -i δ_{a,j} k_j (\widehat{b_iu_j} - \widehat{u_i b_j}).
\end{align}
$$

Such equations form could be solved directly by the MHDFLows packages. Check out ``src\Solver\`` for more details.


To perform a HD/MHD simulation, the workflow of the MHDFlows.jl can break into three parts

1. Problem Declaration/Generation 
2. Initial Condition Set Up
3. Time integration 

Check out other section for the set up of each parts. For a full set up and workflow of a simulation, check out the jupyter notebook example. 