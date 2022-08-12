
from  ase import io
from aiida.orm import Str, Dict, Float, Int
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import calcfunction, workfunction, submit, run


# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')

import numpy as np
from ase import Atoms
from ase.data import atomic_masses

def ase_atoms_supply_Z_from_mass(atoms):
    def _find_Z_sim_mass(M, eps=1e-2):
        for i,m in enumerate(atomic_masses):
            if np.abs(m-M)<eps:
                return i
        raise ValueError('failed to find M')
        return None
    Z = []
    for i,m in zip(atoms.get_atomic_numbers(), atoms.get_masses()):
        z = _find_Z_sim_mass(m)
        Z.append(z)
    newatoms = Atoms(numbers=Z, scaled_positions=atoms.get_scaled_positions(),
                     masses=atoms.get_masses(), cell=atoms.cell, pbc=atoms.pbc)
    return newatoms


import spglib
from ase import Atoms
def get_prim_conv_atoms(atoms):
    cell, scaled_positions, numbers = spglib.find_primitive(
            atoms)
    prim_structure = Atoms(cell=cell, scaled_positions=scaled_positions,
                           numbers=numbers)

    cell, scaled_positions, numbers = spglib.standardize_cell(
            prim_structure)
    conv_structure = Atoms(cell=cell, scaled_positions=scaled_positions,
                           numbers=numbers)

    return prim_structure, conv_structure



def load_atoms_bare(filename: str, format: str, style=None, supply_Z_from_mass=None) -> Atoms:
    """load atom from file, filename
    
    style must be supplied if format=="lammps-data".

    supply_Z_from_mass is valid only when format=="lammps-data".
    
    Args:
        filename (str): filename to read.
        format (str): filename format.
        style (str): style option to read format. Defaults to None.
        supply_Z_from_mass (bool): to supply mass or not. Defaults to None.
        
    Returns:
        ase.Atoms: cystal data.
    """
    if format == "cif":
        cif = CifData(file=filename)
        atoms = cif.ase
    elif format == "lammps-data":
        #atoms = read_lammps_data(filename, style=style)
        atoms = io.read(filename, format=format, style=style)
        if supply_Z_from_mass is not None:
            if supply_Z_from_mass:
                atoms = ase_atoms_supply_Z_from_mass(atoms)
    elif format== "general":
        atoms = io.read(filename, format=format)
    else:
        print("unknown format", format)
        atoms = io.read(filename, format=format)
    return atoms



@calcfunction
def load_atoms(filename: Str, format: Str, style=None, supply_Z_from_mass=None) -> StructureData:
    """load atom from file, filename
    
    style must be supplied if format=="lammps-data"
    
    Args:
        filename (Str): filename to read.
        format (Str): filename format.
        style (Str): style option to read format. Defaults to None.
        supply_Z_from_mass (Bool): to supply mass or not. Defaults to None.

    Returns:
        StructureData: cystal data.
    """
    format = format.value
    filename = filename.value

    if style is not None:
        style = style.value
    if supply_Z_from_mass is not None:
        supply_Z_from_mass = supply_Z_from_mass.value
    atoms = load_atoms_bare(filename, format, style, supply_Z_from_mass)

    return StructureData(ase=atoms)

