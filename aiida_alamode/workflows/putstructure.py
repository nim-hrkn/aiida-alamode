

from aiida.engine import WorkChain
from aiida.engine import calcfunction
from aiida.plugins import DataFactory
from aiida.orm import Str, Float
import os
from .structure_tools import make_atoms_primcell, make_atoms_supercell, \
    structure_to_SinglefileData, make_atoms_standardizedcell

# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')

@calcfunction
def _path_join(cwd: Str, template: Str, value: Str):
    filename = template.value.replace("{format}", value.value)
    return Str(os.path.join(cwd.value, filename))


class PutStructuresWorkChain(WorkChain):
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
