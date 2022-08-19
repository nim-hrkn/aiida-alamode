
import spglib
from ase.data import atomic_masses
from ase import Atoms
import numpy as np
from ase import io
from aiida.orm import Str, Dict, Float, Int
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import calcfunction, workfunction, submit, run


# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')


def ase_atoms_supply_Z_from_mass(atoms):
    def _find_Z_sim_mass(M, eps=1e-2):
        for i, m in enumerate(atomic_masses):
            if np.abs(m-M) < eps:
                return i
        raise ValueError('failed to find M')
        return None
    Z = []
    for i, m in zip(atoms.get_atomic_numbers(), atoms.get_masses()):
        z = _find_Z_sim_mass(m)
        Z.append(z)
    newatoms = Atoms(numbers=Z, scaled_positions=atoms.get_scaled_positions(),
                     masses=atoms.get_masses(), cell=atoms.cell, pbc=atoms.pbc)
    return newatoms


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


def load_atoms_bare(filename: str, format: str, supply_Z_from_mass=True) -> Atoms:
    """load atom from file, filename

    style must be supplied if format=="lammps-data".

    supply_Z_from_mass is valid only when format=="lammps-data".

    Args:
        filename (str): filename to read.
        format (str): filename format.
        supply_Z_from_mass (bool): to supply mass or not. Defaults to True.

    Returns:
        ase.Atoms: cystal data.
    """
    if format == "lammps-data":
        #atoms = read_lammps_data(filename, style=style)
        style = 'atomic'
        supply_Z_from_mass = True
        atoms = io.read(filename, format=format, style=style)
        if supply_Z_from_mass:
            atoms = ase_atoms_supply_Z_from_mass(atoms)
    else:
        atoms = io.read(filename, format=format)

    return atoms


@calcfunction
def load_atoms(filename: Str, format: Str) -> StructureData:
    """load atom from file, filename

    style must be supplied if format=="lammps-data"

    Args:
        filename (Str): filename to read.
        format (Str): filename format.

    Returns:
        StructureData: cystal data.
    """
    format = format.value
    filename = filename.value

    atoms = load_atoms_bare(filename, format)

    return StructureData(ase=atoms)
