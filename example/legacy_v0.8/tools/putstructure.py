

from aiida.engine import WorkChain
from aiida.engine import calcfunction

from aiida.plugins import DataFactory


from aiida.orm import Str, Float


# from alamode_aiida.io import load_anphon_kl, load_anphon_kl_spec

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
                                                              to_primitive=False,
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


@calcfunction
def _path_join(cwd: Str, template: Str, value: Str):
    filename = template.value.replace("{format}", value.value)
    return Str(os.path.join(cwd.value, filename))


class PutStructure(WorkChain):
    """structure_filenameで指定された構造をprimicell, supercellのファイルに直す。
    長周期はfactorで指定する。

    AiiDAのStructureData, SinglefileDataがファイルのpathが得られないので用いない。
    すべてStrのフィル名をcwdディレクトリに出力する。
    """
    _CWD = ""
    _PRIM_FILENAME = "primcell.{format}"
    _STANDARDIZED_FILENAME = "standardizedcell.{format}"
    _SUPER_FILENAME = "supercell.{format}"
    _SYMPREC = 1e-5

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str, default=lambda: Str(cls._CWD))
        spec.input("structure", valid_type=StructureData)
        spec.input("factor", valid_type=ArrayData)
        spec.input("format", valid_type=Str)
        spec.input("symprec", valid_type=Float,
                   default=lambda: Float(cls._SYMPREC))

        spec.outline(cls.make_cwd,
                     cls.put_primcell,
                     cls.put_standarizedcell,
                     cls.put_supercell)

        spec.output("primstructure", valid_type=StructureData)
        spec.output("standardizedstructure", valid_type=StructureData)
        spec.output("superstructure", valid_type=StructureData)

        spec.output("primstructure_file", valid_type=SinglefileData)
        spec.output("standardizedstructure_file", valid_type=SinglefileData)
        spec.output("superstructure_file", valid_type=SinglefileData)

    def make_cwd(self):
        if len(self.inputs.cwd.value) > 0:
            os.makedirs(self.inputs.cwd.value, exist_ok=True)

    def put_primcell(self):
        primstructure = make_atoms_primcell(
            atoms=self.inputs.structure, symprec=self.inputs.symprec)
        self.out('primstructure', primstructure)

        if len(self.inputs.cwd.value) > 0:
            filepath = _path_join(self.inputs.cwd, Str(
                self._PRIM_FILENAME), self.inputs.format)
            primstructure_file = structure_to_SinglefileData(atoms=primstructure,
                                                             path=filepath,
                                                             format=self.inputs.format)
            self.out("primstructure_file", primstructure_file)

    def put_standarizedcell(self):
        standardizedstructure = make_atoms_standardizedcell(
            atoms=self.inputs.structure, symprec=self.inputs.symprec)
        self.out('standardizedstructure', standardizedstructure)
        self.ctx.standardizedstructure = standardizedstructure

        if len(self.inputs.cwd.value) > 0:
            filepath = _path_join(self.inputs.cwd, Str(
                self._STANDARDIZED_FILENAME), self.inputs.format)
            superstructure_file = structure_to_SinglefileData(atoms=standardizedstructure,
                                                              path=filepath,
                                                              format=self.inputs.format)
            self.out("standardizedstructure_file", superstructure_file)

    def put_supercell(self):
        superstructure = \
            make_atoms_supercell(atoms=self.ctx.standardizedstructure,
                                 factor=self.inputs.factor)
        self.out('superstructure', superstructure)

        if len(self.inputs.cwd.value) > 0:
            filepath = _path_join(self.inputs.cwd, Str(
                self._SUPER_FILENAME), self.inputs.format)
            superstructure_file = structure_to_SinglefileData(atoms=superstructure,
                                                              path=filepath,
                                                              format=self.inputs.format)

            self.out("superstructure_file", superstructure_file)
