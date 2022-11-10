# MHDFlows.jl

## Introduction
Three Dimensional Magnetohydrodynamic(MHD) simulation packages natively written in Julia language with the use of [FourierFlows.jl](http://github.com/FourierFlows/FourierFlows.jl). The simulation is based on pseudo-spectral method. This solver support the following features:

1. 2D incompressible HD/MHD simulation (periodic boundary)
2. 3D incompressible HD/MHD simulation (periodic boundary)
3. Incompressible  HD/MHD simulation with volume penalization method
4. Passive Dye Tracer (Experimental Feature)

This package leverages the [FourierFlows.jl](http://github.com/FourierFlows/FourierFlows.jl) package to set up the module. The main purpose of MHDFlows.jl aims to solve the portable 3D MHD problems on personal computer instead of cluster. With native GPU accerlation support, the MHDFlows.jl could solve the front-end MHD turbulence research problems in the order of few-ten minutes by using a mid to high end gaming display card (see Memory usage & speed section). Feel free to modify yourself for your own research purpose.  

## Examples
Few jupyter notebook examples were set up to illustrate the workflow of using the package. See `example\` for more detail.  The documentation is work in progress and will be available in the future. 

## Developer
MHDFlows is currently developed by [Ka Wai HO@UW-Madison Astronomy](https://scholar.google.com/citations?user=h2j8wbYAAAAJ&hl=en).

## Cite
A paper can be cited elsewhere in the future. Feel free to cite the GitHub page right now. 
