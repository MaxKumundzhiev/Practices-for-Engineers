## 3D visualization

This lecture will cover two aspects of this topic
1. File formats that support efficient storage of large datasets
2. Jupyter modules and independent softwares that are used for visualization

### 1. File formats
Historically many softwares that produced a large amount of output created its own file format. Softwares doing classical molecular dynamics, density functional theory calculations, Quantum Monte Carlo etc not only produce data for the user as an end result, but also generates data that are reused during iterative calculations or saved as checkpoints in case of failure. These data are/were stored in custom defined binary files first and later with the spread of standardised file formats such as cube, hdf5, netcdf etc. data became much easy to transfer.

#### Hierarchical Data Format (**HDF5**) -
Useful links:
* https://www.hdfgroup.org/solutions/hdf5/
* Numpy npz versus hdf5: https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr
* Why not to use hdf5: https://cyrille.rossant.net/moving-away-hdf5/
* About B-trees: https://www.youtube.com/watch?v=aZjYr87r1b8

#### The Network Common Data Form (**netcdf**) 
https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html


### Visualization packages

* Ipyvolume: 
https://ipyvolume.readthedocs.io/en/latest/
  * IPyvolume is a Python library to visualize 3d volumes and glyphs (e.g. 3d scatter plots), in the Jupyter notebook, with minimal configuration and effort
* Visit:
https://wci.llnl.gov/simulation/computer-codes/visit/downloads
  * VisIt is an Open Source, interactive, scalable, visualization, animation and analysis tool.
* Paraview: https://www.paraview.org/
* VMD: https://www.ks.uiuc.edu/Research/vmd/
