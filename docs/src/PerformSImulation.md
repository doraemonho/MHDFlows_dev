# Perform Simulation/ Time Intregation


Performing the time integration is done using the ``TimeIntegrator!`` function. It controls the 

1. time intravel/ number of iterations for the integration
2. timestep setup (CFL condition/ user defined time step/ user defined CFL function)
3. file saving option and saving time intravel
4. diagnosis function
5. dashboard of displaying the integraion information. (dynamic/static)

The input format are 
```
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

``t₀,N₀`` are the total time intavel and the total iterations. 
The simulation will stop either one of the condition has been satisfied.