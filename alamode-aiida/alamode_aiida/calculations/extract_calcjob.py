
from aiida.orm import Code
from aiida.orm import Str, Int, Dict, List, Float
from aiida.engine import calcfunction, WorkChain, ToContext, append_
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.common.folders import Folder
from aiida.parsers.parser import Parser
from aiida.engine import CalcJob, calcfunction, WorkChain
from aiida.common.datastructures import CalcInfo, CodeInfo

from itertools import cycle
import os

from aiida.engine import calcfunction, workfunction, submit, run
from aiida.engine import run_get_node
from aiida.orm import load_code, load_node
from ase import io
from ..io.lammps_support import write_lammps_data
from ..io.ase_support import load_atoms
from ..io.aiida_support import save_output_folder_files

# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
TrajectoryData = DataFactory('array.trajectory')


class extractCalcJob(CalcJob):
    """ extract.py

    Specify offset = SinglefileData if there is. offset = Str means no file.
    You can add additional options by 'options' as List, where the files are assumed to be in the cwd directory.
    """
    _WITHMPI = True
    _CWD = ""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("format", valid_type=Str, help='structure file format')
        spec.input("structure_org", valid_type=StructureData,
                   help='equilibrium structure')
        # spec.input("structure_org_filename", valid_type=Str)
        spec.input("displacement_and_forces", valid_type=Dict,
                   help='displacement and focrces')
        spec.input("cwd", valid_type=Str,
                   default=lambda: Str(cls._CWD), help='directory where results are saved.')
        spec.input("prefix", valid_type=Str, help='string added to filenames')
        spec.input("offset", valid_type=(Str, SinglefileData), default=lambda: Str(""),
                   help='offset force file.')
        spec.input("options", valid_type=List, default=lambda: List(list=[]),
                   help='additional options, such as --unit, --get, ... without any filenames.')

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.extract'
        spec.inputs['metadata']['options']['input_filename'].default = 'extract.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'DFSET'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('dfset', valid_type=List)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        cwd = self.inputs.cwd.value

        displacement_and_forces = list(
            self.inputs.displacement_and_forces.get_dict().items())
        if len(displacement_and_forces) != 1:
            raise ValueError('displacement_and_forces must be len 1.')
        format_, displacement_and_forces_list = displacement_and_forces[0]
        filename_list = []
        for _i, _content in enumerate(displacement_and_forces_list):
            filename = f"{_i}.in"
            filename_list.append(filename)
            with folder.open(filename, "w", encoding='utf8') as handle:
                handle.write(_content)

        atoms = self.inputs.structure_org.get_ase()

        structure_org_filename = "structure_org.in"
        if format_ == "LAMMPS":
            style = 'atomic'
            with folder.open(structure_org_filename, 'w', encoding='utf8') as handle:
                write_lammps_data(
                    handle, atoms, atom_style=style, force_skew=True)
        elif format_ == "QE":
            with folder.open(structure_org_filename, 'w', encoding='utf8') as handle:
                io.write(handle, style="espresso-in")
        elif format_ == "VASP":
            with folder.open(structure_org_filename, 'w', encoding='utf8') as handle:
                io.write(handle, style="vasp")
        else:
            raise ValueError(f'unknown format. format={self.format.value}')

        offset_file = self.inputs.offset
        if isinstance(offset_file, SinglefileData):
            _content = offset_file.get_object_content()
            filename = offset_file.list_object_names()[0]
            with folder.open(filename, "w", encoding='utf8') as handle:
                f.write(_content)

        # code
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [f"--{format_}", f"{structure_org_filename}"]
        if isinstance(offset_file, SinglefileData):
            _filename = offset_file.list_object_names()[0]
            codeinfo.cmdline_params.extend(["--offset", _filename])
        options = self.inputs.options.get_list()
        if len(options) > 0:
            codeinfo.cmdline_params.extend(options)
        codeinfo.cmdline_params.extend(filename_list)

        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = self.options.withmpi

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]

        # add files to retrieve list
        pattern_files = [self.options.output_filename,
                         "_aiidasubmit.sh", structure_org_filename]
        pattern_files.extend(filename_list)
        if isinstance(offset_file, SinglefileData):
            filename = offset_file.list_object_names()[0]
            pattern_files.append(filename)

        calcinfo.retrieve_list = pattern_files

        return calcinfo


class extract_ParseJob(Parser):

    def parse(self, **kwargs):

        cwd = self.node.inputs.cwd.value
        prefix = self.node.inputs.prefix.value

        if len(cwd) > 0:
            os.makedirs(cwd, exist_ok=True)

        try:
            output_folder = self.retrieved
        except:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        if False:
            if len(cwd) > 0:
                for filename in output_folder.list_object_names():
                    _content = output_folder.get_object_content(filename)
                    target_path = os.path.join(cwd, filename)
                    with open(target_path, "w") as f:
                        f.write(_content)
        else:
            conversion_table = save_output_folder_files(output_folder, cwd, prefix)

        filename = self.node.get_option('output_filename')
        _content = output_folder.get_object_content(filename).splitlines()

        self.out('dfset', List(list=_content))
