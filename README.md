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

0. * Install git

      `sudo apt install git`
   
   * Install virtualenv
   
      `sudo apt install python3-virtualenv`
    
1. Create working directory 

    `mkdir fem2geo`
    
    `cd fem2geo`

2. Clone repository from `https://github.com/pabloitu/fem2geo` into created directory.
  
    `clone git https://github.com/pabloitu/fem2geo`
  or download .tar.gz and extract in selected folder
  
3. Create virtual environment for installation

    `cd csep2`  
    `mkdir venv`  
    `cd venv`  
    `python3 -m venv csep-dev`  
    `source csep-dev/bin/activate`  
    `cd ..`  
    `pip3 install numpy` (Because of obspy and scipy)  
    `pip3 install wheel`  
    `pip3 install -r requirements.txt`
    
    Note: If you want to go back to your default environment use the command `deactivate`.
    
    Note: There is an issue installing Cartopy on MacOS with Proj >=6.0.0 and will be addressed in 0.18 release of Cartopy. 
    If this package is needed please manually install or use Conda instructions above. Additionally, if you choose the 
    manual build, you might need to resolve build issues as they arise. This is usually caused by [not having the proper 
    python statics installed](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory) to build the binary packages or poorly written setup.py scripts from other packages.
    
    Also python 3.7 is required.
    
