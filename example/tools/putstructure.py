
from aiida.manage.configuration import get_profile

from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder

from aiida.engine import CalcJob, WorkChain
from aiida.engine import calcfunction, workfunction, submit, run

from aiida.plugins import DataFactory, WorkflowFactory

from aiida.parsers.parser import Parser
from aiida.orm import Code
from aiida.orm import load_code, load_node
from aiida.orm import Str, Dict, Float, Int
from aiida.orm import QueryBuilder

import aiida

from alamode_aiida.data_loader import load_anphon_kl, load_anphon_kl_spec
from ase.io.espresso import write_espresso_in
from ase.io.lammpsdata import read_lammps_data
from ase.build import make_supercell
from ase import Atom, Atoms
import ase

from itertools import combinations_with_replacement
import numpy as np
import os
import subprocess
import shutil
import re
from time import sleep
import spglib


from os.path import expanduser

from .nodebank import NodeBank
from .aiida_support import wait_for_node_finished
from .ase_support import load_atoms

from .lammps_support import write_lammps_data


# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')


@calcfunction
def make_atoms_supercell(atoms: StructureData,
                         diag: ArrayData,
                         ) -> StructureData:
    """diag倍のsupercellをつくる。

    """
    atoms = atoms.get_ase()
    print("atoms", atoms)
    # supercell
    _diag = diag.get_array('diag')
    P = np.diag(_diag)  # define supercell.
    print("P", P)
    super_structure = make_supercell(atoms, P=P)  # ase function
    super_structure = StructureData(ase=super_structure)
    print("super", super_structure.get_ase())
    return super_structure


@calcfunction
def make_atoms_primcell(atoms: StructureData
                        ) -> StructureData:
    """primitive cell構造をつくる。

    """
    atoms = atoms.get_ase()
    # primitive cell
    cell, scaled_positions, numbers = spglib.find_primitive(
        atoms)
    prim_structure = Atoms(cell=cell, scaled_positions=scaled_positions,
                           numbers=numbers)
    prim_structure = StructureData(ase=prim_structure)
    print("prim", prim_structure.get_ase())
    return prim_structure


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
def join_path(dir: Str, name: Str):
    path = os.path.join(dir.value, name.value)
    return Str(path)


class PutStructure(WorkChain):
    """structure_filenameで指定された構造をprimicell, supercellのファイルに直す。
    長周期はdiagで指定する。

    AiiDAのStructureData, SinglefileDataがファイルのpathが得られないので用いない。
    すべてStrのフィル名をcwdディレクトリに出力する。
    """
    _PRIM_FILENAME = "primcell.dat"
    _SUPER_FILENAME = "supercell.dat"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str)
        spec.input("structure", valid_type=StructureData)
        spec.input("diag", valid_type=ArrayData)
        spec.input("output_format", valid_type=Str)
        spec.input("primcell_filename", valid_type=Str,
                   default=lambda: Str(cls._PRIM_FILENAME))
        spec.input("supercell_filename", valid_type=Str,
                   default=lambda: Str(cls._SUPER_FILENAME))
        spec.outline(cls.make_path,
                     cls.put_primcell,
                     cls.put_supercell)
        spec.output("primstructure_filepath", valid_type=Str)
        spec.output("superstructure_filepath", valid_type=Str)
        spec.output("primstructure", valid_type=StructureData)
        spec.output("superstructure", valid_type=StructureData)
        spec.output("primstructure_file", valid_type=SinglefileData)
        spec.output("superstructure_file", valid_type=SinglefileData)

    def make_path(self):
        superstructure_filepath = join_path(self.inputs.cwd,
                                            self.inputs.supercell_filename)
        self.ctx.superstructure_filepath = superstructure_filepath
        primstructure_filepath = join_path(self.inputs.cwd,
                                           self.inputs.primcell_filename)
        self.ctx.primstructure_filepath = primstructure_filepath
        self.out("primstructure_filepath", self.ctx.primstructure_filepath)
        self.out("superstructure_filepath", self.ctx.superstructure_filepath)

    def put_primcell(self):
        primstructure = make_atoms_primcell(atoms=self.inputs.structure)
        self.out('primstructure', primstructure)

        primstructure_file = structure_to_SinglefileData(atoms=primstructure,
                                                         path=self.ctx.primstructure_filepath,
                                                         format=self.inputs.output_format)
        self.out("primstructure_file", primstructure_file)

    def put_supercell(self):
        superstructure = \
            make_atoms_supercell(atoms=self.inputs.structure,
                                 diag=self.inputs.diag)
        self.out('superstructure', superstructure)
        superstructure_file = structure_to_SinglefileData(atoms=superstructure,
                                                          path=self.ctx.superstructure_filepath,
                                                          format=self.inputs.output_format)

        self.out("superstructure_file", superstructure_file)
