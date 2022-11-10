# Initial Condition (IC)

To set up the initial condition for the MHDFlows problems, we will use the ``SetUpProblemIC`` function.

The input format of the function are : 
```
SetUpProblemIC!(prob; ux = [], uy = [], uz =[],
                      bx = [], by = [], bz =[],
                      U₀x= [], U₀y= [], U₀z=[],
                      B₀x= [], B₀y= [], B₀z=[])
```
``ux/uy/uz/bx...`` are the IC of the physics variables in real space.
``U₀x/U₀y/...`` are the solid domain for Volume penalization method.
by default, only the input of MHDFlows Problem ``prob`` is mandatory, other parameters are optional. The IC will be zeros if user doesnt declare in the ``SetUpProblemIC!``.

An example of a IC :
````
Nx = Ny = Nz = 128;
T  = Float32;
ux = ones(T,(Nx,Ny,Nz));
# only ux will be 1 and other variable will be zero.
SetUpProblemIC!(prob; ux = ux);
````


!!! hint
If user prefers set up the initial condition in spectral space, consider copy the spectral IC to ``prob.sol``.