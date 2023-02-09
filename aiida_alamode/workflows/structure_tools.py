

from aiida.engine import WorkChain
from aiida.engine import calcfunction

from aiida.plugins import DataFactory


from aiida.orm import Str, Float


from ase.build import make_supercell
from ase import Atoms
import ase

import os
import spglib


from aiida_alamode.io import write_lammps_data


# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')


@calcfunction
def make_atoms_supercell(atoms: StructureData,
                         factor: ArrayData,
                         ) -> StructureData:
    """factor倍のsupercellをつくる。

    """
    atoms = atoms.get_ase()
    print("atoms", atoms)
    # supercell
    P = factor.get_array('factor')

    super_structure = make_supercell(atoms, P=P)  # ase function
    super_structure = StructureData(ase=super_structure)
    return super_structure


@calcfunction
def make_atoms_primcell(atoms: StructureData, symprec: Float
                        ) -> StructureData:
    """primitive cell構造をつくる。

    """
    atoms = atoms.get_ase()
    symprec = symprec.value
    # primitive cell
    cell, scaled_positions, numbers = spglib.find_primitive(
        atoms, symprec=symprec)
    prim_structure = Atoms(cell=cell, scaled_positions=scaled_positions,
                           numbers=numbers)
    prim_structure = StructureData(ase=prim_structure)
    print("prim", prim_structure.get_ase())
    return prim_structure


@calcfunction
def make_atoms_standardizedcell(atoms: StructureData, symprec: Float
                                ) -> StructureData:
    """standardized cell構造をつくる。

    """
    atoms = atoms.get_ase()
    symprec = symprec.value
    # primitive cell
    cell, scaled_positions, numbers = spglib.standardize_cell(atoms,
                                                              to_primitive=True,
                                                              no_idealize=False,
                                                              symprec=symprec)

    standardized_structure = Atoms(cell=cell, scaled_positions=scaled_positions,
                                   numbers=numbers)
    standardized_structure = StructureData(ase=standardized_structure)
    print("standardized", standardized_structure.get_ase())
    return standardized_structure


@calcfunction
def structure_to_SinglefileData(atoms: StructureData,
                                path: Str,
                                format: Str) -> SinglefileData:
    """AiiDAのStructureDataからSinglefileDataをつくる。
    """
    atoms = atoms.get_ase()
    print("atoms", atoms)
    print("path", path)
    print("format", format)
    target_path = path.value
    if format == "lammps-data":
        with open(target_path, "w") as fd:
            write_lammps_data(fd, atoms, force_skew=True)
    else:
        ase.io.write(target_path, atoms, format=format.value)

    prim_file = SinglefileData(target_path)
    return prim_file


