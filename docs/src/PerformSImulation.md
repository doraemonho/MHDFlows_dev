# Performing Simulation

## Time Intregation

Performing the time integration is done using the `TimeIntegrator!` function. It controls the 

1. time intravel/ number of iterations for the integration
2. timestep setup (CFL condition/ user defined time step/ user defined CFL function)
3. file saving option and saving time intravel
4. diagnosis function
5. dashboard of displaying the integraion information. (dynamic/static)

The input format are 
```julia
TimeIntegrator!(prob,t₀,N₀;
               usr_dt = 0.0,
             CFL_Coef = 0.25,
         CFL_function = nothingfunction,
                diags = [],
    dynamic_dashboard = true,
          loop_number = 100,
                 save = false,
             save_loc = "",
             filename = "",
          file_number = 0,
              dump_dt = 0)
```

`t₀/N₀` are the total time intavel and the total iterations. 
The simulation will stop either one of the condition has been reached.

## File saving & reading format
If the save flag `save = true`, the saving function would turned on and write an output file after every `dump_dt`. The file location would be saved under `save_loc*file_loc`. For example,

```julia
save_loc = "/home/user/"
filename = "TestingA_"
``` 
The fisrt write file would be store as
```
/home/user/TestingA_t_0001.h5
```
The format of file are store as HDF5. The structure are 
```
TestingA_t_0001.h5
├── t
├── i_velocity
├── j_velocity
├── k_velocity
├── i_mag_field
├── j_mag_field
├── k_mag_field
└── dye_density
```
The below code provide an example for user to access certain object.

```julia
using HDF5

file_path = "/home/user/TestingA_t_0001.h5";
f = h5open(file_path,"r");
iv = read(f,"i_velocity");
```
Alternatively, user could also use the `ReadMHDFLows` function (not including the dye density access at the moment) as:

```julia
using HDF5
iv,jv,kv,ib,jb,kb,t = readMHDFlows(file_path)
```
