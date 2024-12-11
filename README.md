# fem2geo

## Description

A simple library that aims to perform traditional structural geology analyses on data from finite element
models (FEM) or boundary element models (BEM), such as visualizing stress orientations on stereo-plots, or providing 
slip/dilation tendency analysis.  The inputs are (for now) ``.vtk`` or ``.vtu`` files from the Adeli 3D 
FEM program.

It consists in two parts:
1. A set of functions to handle a ``vtk`` results file from a FEM model, which obtain the data
from region of interests (e.g., a sphere, a box). This step can be skipped and performed 
a-priori in paraview (e.g. threshold the model by a value and saved as a new mesh).

2. A set of functions that uses the scalar/vector/tensor data from the ``.vtk``, and calculates
geological variables of interests, which can be then mapped onto a new ``.vtk`` or plotted in 
stereomaps.

## Installation and setup

### Using pip

To use`pip`, it is required to have installed a `python` version between 3.9 and 3.11, and the following the dependencies.

1. Installing basic dependencies (Linux):

    ```shell
    sudo apt install git python3-pip python3-virtualenv
    ```

2. Clone (or download) package

    ```shell
    git clone https://github.com/pabloitu/fem2geo
    cd fem2geo
    ```
   
   (Update the package with `git pull`)
   
3. Create a virtual environment 

   ```shell
   python3 -m venv venv
   ```
   This creates a `venv` folder that contains the environment's local packages, etc.

4. Activate the environment with:

   ```
   source venv/bin/activate
   ```
    
   * Note: To deactivate virtual environment when desired, type ```deactivate```

5. Install ``fem2geo`` package and its dependencies with:

   ```shell
   pip install -e .
   ```
   
    * [Optional] To run the codes using jupyter-lab/notebooks in the virtual environment:
    
       `pip install jupyterlab`    

      or with `spyder`: 

       `pip install spyuder`

6. Run an example from the ```examples``` folder with:

   ```shell
   cd examples
   python3 A_principal_directions.py
   ```

   * [Optional] To run the codes with jupyterlab or spyder, type `juypyterlab`/`spyder` in the console.


### Using Anaconda/conda

1. Install ``conda``. Check https://conda.io . Anaconda is recommended to have a GUI for python with Jupyterlab or spyder)
2. Create conda environment
   ```
   conda create -n fem2geo python=3.10
   conda activate fem2geo
   ```
3. Clone and install ``fem2geo`` package
    ```
    git clone https://github.com/pabloitu/fem2geo
    cd fem2geo
    pip install -e .
    ```
   
    * [Optional] To run the codes using jupyter-lab/notebooks in the virtual environment:
    
       `pip install jupyterlab`
   
4. Run example

    `python ex1_subset.py`

   * [Optional] To run the codes with jupyterlab, open `juypyter-lab` from the console.





