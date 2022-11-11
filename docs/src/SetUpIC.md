# Initial Condition (IC)


## Set up the initial condition
To set up the initial condition for the MHDFlows problems, we will use the `SetUpProblemIC` function.

The input format of the function are : 
```julia
SetUpProblemIC!(prob; ux = [], uy = [], uz =[],
                      bx = [], by = [], bz =[],
                      U₀x= [], U₀y= [], U₀z=[],
                      B₀x= [], B₀y= [], B₀z=[])
```
`U₀x/U₀y/...` are the solid domain for Volume penalization method.
by default, only the input of MHDFlows Problem `prob` is mandatory, other parameters are optional. The IC will be zeros if user doesnt declare in the `SetUpProblemIC!`.

An example of a IC :
````julia
Nx = Ny = Nz = 128;
T  = Float32;
ux = ones(T,(Nx,Ny,Nz));
# only ux will be 1 and other variable will be zero.
SetUpProblemIC!(prob; ux = ux);
````

!!! hint
	If user prefers set up the initial condition in spectral space, consider copy the spectral IC to `prob.sol`.


## Restart
For restarting the simulation from the saved file, user could use `Restart!` function. The function would handle all the set up of restoring proess.

To restart the simulation, user should declare the problem first and following the below example
```julia
file_path_and_name = "/home/user/Testing_A_t_0001.h5"
Restart!(prob,file_path_and_name)
```

!!! warning
    User should be careful that the format of the data and problem are the same! (i.e., same size for both file and problem/ file should contain b-field if user is declaring a MHD problem)