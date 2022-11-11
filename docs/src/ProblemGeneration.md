# Probelm Generation/ Problem Declaration

At the step of problem declaration, you declare the key feature of the simulation, such as :
1. computation architecture and its format (`CPU()`/`GPU()` or `Float32`/`Float64`)
2. types of the problem (HD/MHD)
3. numerical parameters (2D/3D, time integator)
4. physical parameter controls the flows (viscosity $\nu/\eta$)
5. feature that you want to include (Force function/ Volume penalization method/ Dye tracing)
6. user defined functions/ data structure will be used during the simulation 

We use the ` MHDFlows.Problem` function to declare a problem. Below is an example of declaring a 3D HD simulation on GPU


```julia
using MHDFlows
using CUDA

#parameters
N = 128;
Lx = 2π;

Re = 150;
U  = 1;
ν  = U*Lx/Re
η  = ν;

# Declare the problem on CPU/GPU
dev = GPU();
T  = Float32;

Re = 1e4;
L  = Lx;
U  = 1;
ν  = U*L/Re

GPUprob =       Problem(dev;
     # Numerical parameters
                   nx = nx,
                   Lx = Lx,
     # Drag and/or hyper-viscosity for velocity/B-field
                    ν = ν,
                   nν = 0,
                    η = η,
                   nη = 0,
     # Declare if turn on magnetic field, VP method, Dye module
       	      B_field = false,
            VP_method = false,
           Dye_Module = false,
     # Timestepper and equation options
              stepper = "RK4",
     # Float type and dealiasing
                    T = T,
     aliased_fraction = 1/3,
     # User defined params/vars
             usr_vars = A99_var,
           usr_params = [],
             usr_func = [])
```
```
#Output
MHDFlows Problem
  │    Funtions
  │     ├──────── B-field: OFF
  ├─────├────── VP Method: OFF
  │     ├──────────── Dye: OFF
  │     └── user function: OFF
  │                        
  │     Features           
  │     ├─────────── grid: grid (on GPU)
  │     ├───── parameters: params
  │     ├────── variables: vars
  └─────├─── state vector: sol
        ├─────── equation: eqn
        ├────────── clock: clock
        └──── timestepper: RK4TimeStepper
```
The function will return a type object `MHDFlowsProblem`, it contains all the data strcuture will be used in the simulation. It can be break into differents child objects

```
  │     MHDFlows Problems           
  │     ├─────────── grid: grid
  │     ├───── parameters: params
  │     ├────── variables: vars
  └─────├─── state vector: sol
        ├─────── equation: eqn
        ├────────── clock: clock
        ├───userfunction : usr_func
       	├─────────── dye : dye
        └──── timestepper: ts
```
To access/visualize a child object (for example grid) simpily type `GPUprob.grid`. For each child objects,

1. `grid `: contains all the information about the grid (Real Space: `Lx/Ly/Lz/Nx/Ny/Nz...`, Spectral Space:`kr/Krsq...`)
2. `params` : Array indexing/ float type parameters ($\nu/\eta$)
3. `vars`: Physical variables ($\vec{v}/\vec{b}$) in real space
4. `sol `: Physical variables ($\vec{v}/\vec{b}$) in spectral space
5. `clock` : time intragrator infromation ($t/dt/step$)
6. `usr_func` : user defined function to execute after each time step
7. `dye` : Dye modules
8. `ts` : timestepper, constains the sketch array and time integration iterator.

!!! warning 
    User shouldn't change any physical variables ($b/v$) when using `usr_func`. It is for visualization purpose only. To interact with physical variables during the timestep, consider the force function. 