# fem2geo

## Description

A simple library that aims to perform traditional structural geology analyses on data from finite element
models (FEM) or boundary element models (BEM).

Current features are:

* Visualizing stress orientations in stereo-plots
* Creating slip/dilation tendency analysis.

The currently supported inputs aare ``.vtk`` or ``.vtu`` files from the Adeli 3D FEM program.

The code consists in two parts:

1. Parse and handle ``vtk`` files, as well as extracting from region of interests (e.g., a sphere, a box within the model).

2. Post-process the scalar/vector/tensor data and visualize structural-geology variables of interests.

## Installation and setup

### Using pip

1. Installing basic dependencies (Linux):

   The package requires a  `python` version >= 3.9 and the following dependencies:

    ```shell
    sudo apt install git python3-pip python3-virtualenv
    ```

2. Clone (or download) package and access its directory

    ```shell
    git clone https://github.com/pabloitu/fem2geo
    cd fem2geo
    ``` 
   (Update the package to the newest version with `git pull`)

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

6. Run an example from the ```examples``` folder with:

   ```shell
   cd examples
   python3 A_principal_directions.py
   ```

7. [Optional] To run the codes with a python IDE, try [spyder](https://www.spyder-ide.org/) (install with `sudo apt install spyder` and type `spyder` in the console when inside the virtual environment) or the [PyCharm community edition](https://www.jetbrains.com/pycharm/download/?section=linux).


### Using conda

1. Install a ``conda`` [distribution](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). Anaconda is recommended if you use Windows and want to have a GUI/IDE for python with spyder, but [Miniforge](https://conda-forge.org/download/) is recommended for a lightweight Linux installation.

2. Create conda environment:

   If Anaconda was installed:
   ```
   conda create -n fem2geo python=3.12
   conda activate fem2geo
   ```
   If Miniforge was installed, replace `conda` by `mamba`

3. Clone and install ``fem2geo`` package
    ```
    git clone https://github.com/pabloitu/fem2geo
    cd fem2geo
    pip install -e .
    ```
   
4. Run example

    `python examples/A_principal_directions.py`


5. modifications by Muriel March
   updated fem2geo/model_handler.py, E_composite_plot.py and .ptovtuUserDefault file in order to adapt to the new .vtu format (tensor_order of stress field, name, units of stress and coordinates). 



