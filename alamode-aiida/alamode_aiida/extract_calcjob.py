
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
from alamode_aiida.ase_support import load_atoms
from ase import io

# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')
TrajectoryData = DataFactory('array.trajectory')


class extractCalcJob(CalcJob):
    """extract.pyのupdateに対応するためにworkchainからcalcjobに直す。"""
    _WITHMPI = True

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("format", valid_type=Str)
        spec.input("structure_org_filename", valid_type=Str)
        spec.input("target_folder", valid_type=FolderData)
        spec.input("cwd", valid_type=Str)
        spec.input("prefix", valid_type=Str)

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.extract'
        spec.inputs['metadata']['options']['input_filename'].default = 'extract.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'extract.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('dfset_file', valid_type=SinglefileData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        cwd = self.inputs.cwd.value

        structure_org_filename = self.inputs.structure_org_filename.value
        filepath = os.path.join(cwd, structure_org_filename)
        folder.insert_path(filepath, dest_name=structure_org_filename)

        target_folder = self.inputs.target_folder
        for filename in target_folder.list_object_names():
            filepath = os.path.join(cwd, filename)
            folder.insert_path(filepath, dest_name=filename)

        format_ = self.inputs.format.value

        # code
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [f"--{format_}", f"{structure_org_filename}"]
        codeinfo.cmdline_params.extend(target_folder.list_object_names())

        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = self.options.withmpi

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]

        # add files to retrieve list
        pattern_files = [self.options.output_filename]
        calcinfo.retrieve_list = pattern_files

        return calcinfo


class extract_ParseJob(Parser):

    def parse(self, **kwargs):

        cwd = self.node.inputs.cwd.value
        prefix = self.node.inputs.prefix.value

        try:
            output_folder = self.retrieved
        except:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        filelist = [self.node.get_option('output_filename')]
        local_filelist = [f"DFSET_{prefix}"]

        for filename, localfilename in zip(filelist, local_filelist):
            if filename in output_folder.list_object_names():
                _content = output_folder.get_object_content(filename)
                target_path = os.path.join(cwd, localfilename)
                with open(target_path, "w") as f:
                    f.write(_content)
                dfset_file = SinglefileData(target_path)
            else:
                raise ValueError(
                    f"no filename={filename} in retrieved data.")

        self.out('dfset_file', dfset_file)
