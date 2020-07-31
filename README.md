# fem2geo

## Description

The codes in this repository are intended to faciliate the reading of .VTK files from the Adeli 3D FEM program,
and perform structural analysis to be shown in new VTK files or stereoplots.

It consists in 2 parts:
a) A set of functions to handle a .vtk FEM model, which read a .vtk and obtain the data from region of interests.
This could also ignored, and be performed previously in paraview (e.g. threshold the model by a value and saved as a new mesh).
b) A set of functions that uses the scalar/vector/tensor data from the .vtk, and calculates geological variables of interests,
which can be then mapped onto a new vtk, or plotted in stereomaps.


## Installation instructions

0.  * Install pip

      `sudo apt install python3-pip` 
      
    * Install git

      `sudo apt install git`
   
    * Install virtualenv
   
      `sudo apt install python3-virtualenv`
    
1. Create and access working directory 

    `mkdir fem2geo`
    
    `cd fem2geo`

2. Clone repository from `https://github.com/pabloitu/fem2geo` into created directory.
  
    `clone git https://github.com/pabloitu/fem2geo`
    
  or download .tar.gz and extract in selected folder
  
3. Create virtual environment for dependencies installation

    `python3 -m venv fem2geo` 
 
4. Activate virtual environment

    `source fem2geo/bin/activate`
    
  * Note: To deactivate virtual environment and go back to default environment...
  
    `deactivate`
    
5. Install dependencies

    `pip3 install numpy`
    
    `pip3 install matplotlib`
    
    `pip3 install meshio`
    
    `pip3 install pyvista`
    
    `pip3 install mplstereonet`
    
    Note: If you want to go back to your default environment use the command `deactivate`.

