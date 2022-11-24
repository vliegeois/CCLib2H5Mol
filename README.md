# CCLib2H5Mol

## Description of the package

InitH5Mol-from-QM.py is a python script that uses cclib python library (https://cclib.github.io) to read quantum chemistry files.
The various properties obtained with cclib are then written in HDF5 format (https://www.hdfgroup.org).
This format stores the properties in binary but can easily be read from various programing language including python (using h5py, https://docs.h5py.org).
The program HDFView (https://www.hdfgroup.org/downloads/hdfview/) can also be used to look at the data stored inside the binary file.

The hierarchy used to store the information is the one use by DrawMol (https://www.unamur.be/sciences/chimie/drawmol), DrawSpectrum (https://www.unamur.be/sciences/chimie/drawspectrum), DrawVib (https://www.unamur.be/sciences/chimie/drawvib) and DrawProfile (https://www.unamur.be/sciences/chimie/drawprofile) in the so-called h5mol format.
All the molecular properties are stored in so-called atomic units.
Thus, the atomic positions are in bohr, the energy in hartree, the dipole moment in e bohr, ...
The only exception are the masses that are not in atomic unit (where the reference is the mass of electron) by in (unified) atomic mass unit (also called Dalton).

The purposes of this package are:
- to be able to use DrawMol, ... to visualize the molecular properties evaluated by all the quantum chemistry codes supported by cclib through the intermediate step of converting the data to h5mol (HDF5 format).
- to store properties from various sources in a common container that is more compact and still easily readable, specially in python.

## Usage

- For help

> InitH5Mol-from-QM.py -h 

- To convert one or more files

> InitH5Mol-from-QM.py file1 file2 ...

This will give file1.h5mol, file2.h5mol ... 

- flag '-v' print each property that is copied and stored in the h5mol file

- flag '--overwrite' will remove file1.h5mol, ... if they exist.
Without this flag, the program skip any file that already exists.


