#!/usr/bin/env python3
import cclib
from cclib.parser.utils import convertor
import argparse
import numpy
import math
import h5py
from pathlib import Path
from periodictable import elements

#FIXME
# Check for ecd spectrum
# check for atomic basis set 6d
#check nmr

# various operators and their dimension
OPERATORS = {"E":1, "mu":3, "theta":9, "m":3, "r":3, "p":3, "BI":3}
PROPERTIES = {"energy":"E", "dipole":"mu", "pollen":"mu,mu", "quadrupole":"theta", "magneticdipole":"m", "gtenlen":"mu,m", "aten":"mu,theta", "beta":"mu,mu,mu", "gamma":"mu,mu,mu,mu", "magneticshielding":"BI,m"}

multiplicities = {0:1, 1:3, -1:4, -2:5, 2:6, -3:7, 3:10, -4:9, 4:15, -5:11, 5:21, -6:13, 6:28}
# mult2type = {1:0, 3:1, 4:-1, 5:-2, 6:2, 7:-3, 10:3, 9:-4, 15:4, 11:-5, 21:5, 13:-6, 28:6}
shelltypes = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6, "SP": -1}



def getCCLib(file, attribute):
    try:
        return getattr(file, attribute)
    except AttributeError:
        return None

def calc_modes_mw(modes_c, atmasses):
    """
    modes_c is sometimes renormalized mode
    """
    nmodes = modes_c.shape[0]
    modes_mw = modes_c.copy()
    modes_mw.shape = (nmodes, -1)
    for i, atmass in enumerate(atmasses):
        modes_mw[:, 3*i:3*i+3] = modes_mw[:, 3*i:3*i+3] * math.sqrt(atmass)
    # make sure modes_mw is a unitary matrix
    for imode in range(nmodes):
        nc = math.sqrt((modes_mw[imode, :] **2).sum())
        if nc == 0.0:
            raise Exception("The normal mode is empty")
        modes_mw[imode, :] = (1.0/nc) * modes_mw[imode, :]
    return modes_mw

def get_property_names(name):
    """
    param:name:either usual name or operator name of a property
    return:( usual name, operator name + ending _deriv part) of the property
    """
    deriv = ""
    # strip the _deriv or _transition part
    pos = name.find("_deriv")
    if pos != -1:
        deriv = name[pos:]
        name = name[:pos]
    pos = name.find("_transition")
    if pos != -1:
        deriv = name[pos:]
        name = name[:pos]
    # name is a usual name?
    if name in PROPERTIES:
        usname = name
        opname = PROPERTIES[name]
    # name is a operator name?
    elif name in list(PROPERTIES.values()):
        opname = name
        usname = list(PROPERTIES.keys())[list(PROPERTIES.values()).index(name)]
    else:
        opname = usname = None
    return usname, opname, deriv


def get_property_size(name):
    """
    get the dimension of a property
    """
    names = get_property_names(name)
    dim = 1
    # return 1 if operator name is None
    if names[1] is None:
        return dim
    operators = names[1].split(",")
    for o in operators:
        if o in OPERATORS:
            dim *= OPERATORS[o]
        else:
            print("%s is not a valid name for a property"%(name))
            dim = 0
    return dim

def get_rank(size):
    """
    size = 3**rank
    """
    return int(math.log(size) /math.log(3))

def get_order(size, natoms):
    """
    size = (3*Natoms) **order
    """
    return int(math.log(size) /math.log(3*natoms))

def isotopicMassesAndAbundances(atnum):
    element = elements[atnum]
    masses = [(iso.mass, iso.abundance) for iso in element]
    masses.sort(key=lambda x: x[1], reverse=True)
    return masses

def mostAbundantIsotopicMass(atnum):
    masses = isotopicMassesAndAbundances(atnum)
    mostAbundant = masses[0]
    return mostAbundant[0]

class WriteHDF5(object):

    def __init__(self, filename, natoms, verbose=False):
        self.ofile = h5py.File(filename, "a")
        self.natoms = natoms
        self.verbose = verbose

    def close(self):
        self.ofile.close()
        self.natoms = 0

    def writeTitle(self, title):
        try:
            self.ofile.create_dataset("title", data=numpy.array(title, dtype=numpy.string_))
            if self.verbose:
                print("Title written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Title already exists in file")

    def writeAtomicCoordinates(self, atnums, coords):
        coords = convertor(coords, "Angstrom", "bohr")
        try:
            self.ofile.create_dataset("coordinates", data=coords)#, compression="gzip")
            self.ofile.create_dataset("atomicNumbers", data=atnums)
            if self.verbose:
                print("Atomic coordinates and atomic numbers written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Atomic Coordinates already exists in file")

    def writeVibrationalModes(self, modes_c, freqs, atmasses):
        try:
            modes_mw = calc_modes_mw(modes_c, atmasses)
            self.ofile.create_dataset("normalModes/modesMW", data=modes_mw)
            self.ofile.create_dataset("normalModes/freqs", data=freqs)
            self.ofile.create_dataset("normalModes/masses", data=atmasses)
            self.ofile.create_dataset("normalModes/natoms", data=self.natoms)
            self.ofile.create_dataset("normalModes/nmodes", data=len(freqs))
            if self.verbose:
                print("Normal modes written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Vibrational Normal Modes already exists in file")

    def writeBasisSet(self, shells):
        try:
            grp = self.ofile.create_group("basisSet")
            shell2atoms = numpy.array([], dtype=numpy.int32)
            shgrp = grp.create_group("shells")
            ishell = 0
            for (iatom, atomShells) in enumerate(shells):
                shell2atoms = numpy.append(shell2atoms, numpy.ones((len(atomShells)), dtype=numpy.int32) * iatom)
                for shell in atomShells:
                    subshgrp = shgrp.create_group(f"{ishell}")
                    l = shelltypes[shell[0]]
                    values = numpy.array(shell[1])
                    coefficients = values[:, 1]
                    exponents = values[:, 0]
                    subshgrp.create_dataset("coefficients", data=coefficients)
                    subshgrp.create_dataset("exponents", data=exponents)
                    subshgrp.create_dataset("l", data=l)
                    subshgrp.create_dataset("multiplicity", data=multiplicities[l])
                    ishell += 1
            grp.create_dataset("shell2Atoms", data=shell2atoms)
            if self.verbose:
                print("Atmic Basis Set written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Basis Set already exists in file")

    def writeMolecularOrbitals(self, lcaos, energies, homos):
        try:
            grp = self.ofile.create_group("listOfOrbitals")
            isRestricted = len(lcaos) == 1
            nMO = len(energies[0])
            imo = 0
            names = ["Alpha Canonical Orbitals", "Beta Canonical Orbitals"]
            for (lcao, energy, homo) in zip(lcaos, energies, homos):
                occupations = numpy.zeros(nMO)
                nOccElectrons = homo + 1
                if isRestricted:
                    occupations[:nOccElectrons] = numpy.ones((nOccElectrons), dtype=numpy.int32) * 2
                else:
                    occupations[:nOccElectrons] = numpy.ones((nOccElectrons), dtype=numpy.int32)
                subgrp = grp.create_group(f"{imo}")
                subgrp.create_dataset("coefficients", data=lcao)
                subgrp.create_dataset("energies", data=convertor(energy, "eV", "hartree"))
                subgrp.create_dataset("occupations", data=occupations)
                subgrp.create_dataset("spin", data=imo)
                subgrp.create_dataset("name", data=numpy.array(names[imo], dtype=numpy.string_))
            if self.verbose:
                print("Molecular orbitals written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Molecular orbitals already exists in file")

    def writeThermochemistry(self, enthalpy, entropy, freeenergy, pressure, temperature):
        try:
            group = self.ofile.create_group("Thermochemistry")
            group.create_dataset("enthalpy", data=enthalpy)
            group.create_dataset("entropy", data=entropy)
            group.create_dataset("freeEnergy", data=freeenergy)
            group.create_dataset("pressure", data=pressure*101325)# from atm to Pa
            group.create_dataset("temperature", data=temperature)
        except RuntimeError:
            raise Exception("WriteHDF5: Thermochemistry property already exists in file")


    def writeMechProperty(self, prop, name):
        """
        Write mechanical properties in a file
        param prop: the property as a numpy array of dimension(3*natoms), (3*natoms, 3*natoms), ...
        param name: name of the property
        """
        try:
            self.ofile.create_dataset(name, data=prop)#, compression="gzip")
            if self.verbose:
                print(f"Property {name} written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Mechanical property ['%s'] already exists in file"%(name))


    def writePropertyDictionary(self, prop, name):
        """
        Write any property in a file
        param prop: the property as a dictionary with the w as keys
        param name: name of the property from operatos in the response function
        """
        try:
            group = self.ofile.create_group(name)
            for (i, pulsation) in enumerate(prop):
                dset = group.create_dataset("val{:03d}".format(i), data=prop[pulsation])
                dset.attrs["omegas"] = numpy.array(pulsation)
                if self.verbose:
                    print(f"Property {name} for w {pulsation} written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Property ['%s'] already exists in file"%(name))

    def writeProperty(self, prop, name, w=None):
        """
        Write any property in a file
        param prop: the property as a numpy array of dimension(3**rank) or (nparam, 3**rank)
        param name: name of the property from operators in the response function
        param w: tuple with pulsation. None for multipoles
        """
        try:
            dset = self.ofile.create_dataset(name, data=prop)
            if (w is not None) and (len(w) > 0):
                dset.attrs["omegas"] = numpy.array(w)
            if self.verbose:
                    print(f"Property {name} written to file")
        except RuntimeError:
            raise Exception("WriteHDF5: Property ['%s'] already exists in file"%(name))

    def writePropertyTransition(self, prop, nstates, name):
        """
        Write any property transition in a file
        param prop:the property as a numpy array of dimension(nstate*3**rank) or (nparam, nstate*3**rank)
        param name:name of the property from operators in the response function
        """
        size = get_property_size(name)
        natoms = self.natoms
        if size == 0:
            return
        order = int(math.log(prop.size//(size*nstates)) / math.log(3*natoms))
        if order == 0:
            try:
                self.ofile.create_dataset(name, data=prop)
                if self.verbose:
                    print(f"Property {name} written to file")
            except RuntimeError:
                raise Exception("WriteHDF5: Transition Property ['%s'] already exists in file"%(name))
        else:
            raise Exception("WritePropertyTransition not implement for order higher than zero")


parser = argparse.ArgumentParser()
parser.add_argument("inputfiles",
                    type=Path,
                    help="QM input files. All formats supported by cclib",
                    nargs="+")

parser.add_argument("--overwrite",
                    help="Remove the previous h5mol file if it already exists",
                    action="store_true")

parser.add_argument("-v", "--verbose",
                    help="Verbose mode: print the name of each property that is written into file",
                    action="store_true")

args = parser.parse_args()

for inputfile in args.inputfiles:
    file = cclib.io.ccread(inputfile)
    # skip file that does not exist
    if file is None:
        continue
    print(f"File {inputfile} contains {file.natom} atoms")
    outputfile = inputfile.with_suffix(".h5mol")
    # check if outputfile already exists
    if outputfile.is_file(): # skip to avoid overwritten
        if args.overwrite:
            outputfile.unlink()
        else:
            print(f"File {outputfile} already exists. Will not be overwritten!!!. Add --overwrite option.")
            continue
    oFILE = WriteHDF5(filename=outputfile.name, natoms=file.natom, verbose=args.verbose)
    # write the cartesian coordinates in bohr
    oFILE.writeAtomicCoordinates(atnums=file.atomnos, coords=file.atomcoords[-1])
    # write charge and mult
    charge = getCCLib(file, "charge")
    if charge is not None:
        oFILE.writeProperty(charge, "charge")
    mult = getCCLib(file, "mult")
    if mult is not None:
        oFILE.writeProperty(mult, "multiplicity")
    # masses
    masses = getCCLib(file, "atommasses")
    if masses is not None:
        oFILE.writeProperty(masses, "masses")
    # atomic charges
    atomcharges = getCCLib(file, "atomcharges")
    if atomcharges is not None:
        for (k, v) in atomcharges.items():
            # skip summed charges
            if k.endswith("_sum"):
                continue
            oFILE.writeProperty(v, k+"Charges")
    # write energies
    scfEnergy = getCCLib(file, "scfenergies")
    if scfEnergy is not None:
        # the first index is for geom opt index
        # the value is in eV
        scfEnergy = convertor(scfEnergy[-1], "eV", "hartree")
        oFILE.writeProperty(scfEnergy, "scfEnergy")
    # write thermo
    enthalpy = getCCLib(file, "enthalpy")
    entropy = getCCLib(file, "entropy")
    freeenergy = getCCLib(file, "freeenergy")
    temperature = getCCLib(file, "temperature")
    pressure = getCCLib(file, "pressure")
    if enthalpy is not None and entropy is not None and freeenergy is not None and temperature is not None and pressure is not None:
        oFILE.writeThermochemistry(enthalpy, entropy, freeenergy, pressure, temperature)
    # write mechanical properties
    Properties_n = ("grads", "hessian")#, "cubic_forces")
    Properties_names = ("energyGradient", "hessian")#, "cubicForces")
    for prop, name in zip(Properties_n, Properties_names):
        p = getCCLib(file, prop)
        if p is not None:
            natoms = file.natom
            if prop == "grads":
                # the first index is for geom opt index
                p = p.reshape((-1, 3*natoms))
                p = p[-1]
            order = get_order(p.size, natoms)
            if order == 1:
                p = p.ravel()
            else:
                p = p.reshape((-1, 3*natoms))
            if prop == "grads": #in cclib, grads is the negative gradient of the energy with respect to atomic coordinates in atomic units 
                p = -p
            oFILE.writeMechProperty(p, name)

    # write NMR shielding
    nmr = getCCLib(file, "nmrtensors")
    if nmr is not None:
        pass
        # print(nmr)
        # oFILE.writeProperty(nmr, "BI,m")

    # write ao basis
    nbasis = getCCLib(file, "nbasis")
    gbasis = getCCLib(file, "gbasis")
    if gbasis is not None:
        oFILE.writeBasisSet(gbasis)
    # write MO orbitals
    homos = getCCLib(file, "homos")
    if homos is not None:
        nAlphaElectrons = homos[0] + 1
        nBetaElectrons = homos[1] + 1 if len(homos) > 1 else nAlphaElectrons
        oFILE.writeProperty(nAlphaElectrons+nBetaElectrons, "nElectrons")
        oFILE.writeProperty(nAlphaElectrons, "nAlphaElectrons")
        oFILE.writeProperty(nBetaElectrons, "nBetaElectrons")
    mocoeffs = getCCLib(file, "mocoeffs")
    moenergies = getCCLib(file, "moenergies")
    if mocoeffs is not None and moenergies is not None and homos is not None:
        oFILE.writeMolecularOrbitals(mocoeffs, moenergies, homos)
    # write the modes of vibration
    vibdisps = getCCLib(file, "vibdisps")
    vibfreqs = getCCLib(file, "vibfreqs")
    if vibdisps is not None and vibfreqs is not None:
        if masses is None:
            # get the mass of the most abundant isotope
            masses = [mostAbundantIsotopicMass(atnumber) for atnumber in file.atomnos]
        oFILE.writeVibrationalModes(vibdisps, vibfreqs, masses)
    vibirs = getCCLib(file, "vibirs")
    if vibirs is not None:
        oFILE.writeProperty(vibirs, "IRAreaIntensity")
    vibramans = getCCLib(file, "vibramans")
    if vibramans is not None:
        oFILE.writeProperty(vibramans, "RamanAreaIntensity")
    # write the various properties and their Cartesian derivatives
    moments = getCCLib(file, "moments")
    if moments is not None:
        debye = convertor(1, "ebohr", "Debye")
        if len(moments) > 1:
            dipole = moments[1] / debye
            oFILE.writeProperty(dipole, "mu")
        if len(moments) > 2:
            values = moments[2] / debye # xx, xy, xz, yy, yz, zz
            quadrupole = numpy.zeros((3,3))
            quadrupole[numpy.triu_indices(3)] = values
            quadrupole = quadrupole + quadrupole.transpose()
            quadrupole[numpy.diag_indices(3)] = quadrupole[numpy.diag_indices(3)] / 2
            oFILE.writeProperty(quadrupole, "theta")
    polarizabilities = getCCLib(file, "polarizabilities")
    if polarizabilities is not None:
        staticPol = polarizabilities[0]
        oFILE.writePropertyDictionary({(0,): staticPol}, "mu,mu")
    # Properties_names = ("mu", "mu,mu", "mu,m", "mu,theta")
    # Properties_names += ("mu_deriv_c", "m_deriv_c", "mu,mu_deriv_c", "mu,m_deriv_c", "mu,theta_deriv_c")
    # # Properties_names += ("theta", "mu,mu,mu", "mu,mu,mu,mu")
    # for prop in Properties_names:
    #     usname, opname, deriv = get_property_names(prop)
    #     p = getattr(file, "get_"+usname+deriv)()
    #     if isinstance(p, dict):
    #         for key in p:
    #             prop_size = get_property_size(opname)
    #             natoms = file.natom
    #             order = get_order(p[key].size//prop_size, natoms)
    #             rank = get_rank(prop_size)
    #             if order == 0:
    #                 p[key] = p[key].ravel()
    #             else:
    #                 p[key] = p[key].reshape((-1, prop_size))
    #         oFILE.writePropertyDictionary(p, name=prop)
    #     elif isinstance(p, numpy.ndarray):
    #         # reshape the property
    #         prop_size = get_property_size(opname)
    #         natoms = file.natom
    #         order = get_order(p.size//prop_size, natoms)
    #         rank = get_rank(prop_size)
    #         if order == 0:
    #             p = p.ravel()
    #         else:
    #             p = p.reshape((-1, prop_size))
    #         oFILE.writeProperty(p, w=None, name=prop)

    # write the various property transition
    Properties_n = ("etenergies", "etdips", "etmagdips", "etoscs", "etrotast")
    Properties_names = ("E_transition", "mu_transition", "m_transition", "oscillatorStrengths", "rotationalStrengths")
    for prop, name in zip(Properties_n, Properties_names):
        p = getCCLib(file, prop)
        if p is not None:
            if isinstance(p, numpy.ndarray):
                nstates = p.shape[0]
                natoms = file.natom
                prop_size = get_property_size(name)
                order = get_order(p.size//(prop_size*nstates), natoms)
                rank = get_rank(prop_size)
                if name == "E_transition":
                # transform energiers to hartree
                    p = convertor(p, "wavenumber", "hartree")
                # reshape the property
                if order == 0:
                    if prop_size == 1:
                        p = p.reshape((nstates))
                    else:
                        p = p.reshape((nstates, prop_size))
                else:
                    p = p.reshape((nstates, -1, prop_size))
                oFILE.writePropertyTransition(p, p.shape[0], name=name)
    oFILE.close()

# enum CodingKeys: String, CodingKey {
#         case version
#         case coord = "coordinates"
#         case atNumbers = "atomicNumbers"
#         case atLabels = "atomicLabels"
#         case masses
#         case oniomLayer
#         case title
#         case tags
#         case urlData
#         case bonds
#         case hydrogenBonds
#         case bondOrders
#         case origin
#         case hybridization
#         case implicitValence
#         case cubeData
#         case charge
#         case multiplicity
#         case pointGroup
#         case scfEnergy
#         case totalEnergy
#         case pcmInfo
#         case ircCoordinate
#         case nElectrons
#         case nAlphaElectrons
#         case nBetaElectrons
#         case basisSet
#         case listOfOrbitals
#         case alphaMOEnergies
#         case alphaMOCoefficients
#         case alphaMOOccupations
#         case betaMOEnergies
#         case betaMOCoefficients
#         case betaMOOccupations
#         case aoCube = "AOcube"
#         case electronDensity
#         case unitCell
#         case normalModes
#         case localizedModes
#         case energyTransition = "E_transition"
#         case dipole = "mu"
#         case dipoleAtomic
#         case dipoleTransition = "mu_transition"
#         case magneticDipoleTransition = "m_transition"
#         case vibronicFCEnergies
#         case vibronicFCFactors
#         case vibronicFCLabels
#         case magneticShielding = "BI,m"
#         case inducedCurrent
#         case inducedCurrentSignedNorm
#         case NICS
#         case dMudX = "mu_deriv_c"
#         case AAT = "m_deriv_c"
#         case alpha = "mu,mu"
#         case alphaAtomic = "mu,mu_atomic"
#         case gten = "mu,m"
#         case aten = "mu,theta"
#         case beta = "mu,mu,mu"
#         case energyGradient = "energyGradient"
#         case hessian
#         case cubicForces
#         case quarticForces
#         case dAlphadX = "mu,mu_deriv_c"
#         case dGtendX = "mu,m_deriv_c"
#         case dAtendX = "mu,theta_deriv_c"
#         case mullikenCharges
#         case espCharges
#         case hirshfeldCharges
#     }
