# Copyright 2022 Hiori Kino
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
from aiida.orm import Str, Dict, List
from aiida.plugins import DataFactory
from aiida.common.folders import Folder
from aiida.parsers.parser import Parser
from aiida.common.datastructures import CalcInfo, CodeInfo

import os

from aiida.common.exceptions import InputValidationError

from ..io.aiida_support import save_output_folder_files, folder_prepare_object
from ..common.base import AlamodeBaseCalculation


# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
TrajectoryData = DataFactory('array.trajectory')


class ExtractCalculation(AlamodeBaseCalculation):
    """ extract.py

    Specify offset = SinglefileData if there is. offset = Str means no file.
    You can add additional options by 'options' as List, where the files are assumed to be in the cwd directory.

    If 'cwd' is given. The retrieved files will be saved in the directory specified by 'cwd'.

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
        spec.input("cwd", valid_type=Str, required=False,
                   help='directory where results are saved.')
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

        displacement_and_forces = list(
            self.inputs.displacement_and_forces.get_dict().items())
        if len(displacement_and_forces) != 1:
            raise InputValidationError('displacement_and_forces must be len 1.')
        format_, displacement_and_forces_list = displacement_and_forces[0]
        filename_list = []

        for _i, _content in enumerate(displacement_and_forces_list):
            filename = f"{_i}.in"
            filename_list.append(filename)
            with folder.open(filename, "w", encoding='utf8') as handle:
                handle.write(_content)
        # Should I write in the cwd also to examine them before running?

        structure_org_filename = "structure_org.in"

        try:
            folder_prepare_object(folder, self.inputs.structure_org, actions=[StructureData],
                                  filename="structure_org.in", format=format_)
        except ValueError as err:
            raise InputValidationError(str(err))
        except TypeError as err:
            raise InputValidationError(str(err))

        offset_file = self.inputs.offset
        if isinstance(offset_file, SinglefileData):
            _content = offset_file.get_object_content()
            filename = offset_file.filename
            with folder.open(filename, "w", encoding='utf8') as handle:
                handle.write(_content)

        # code
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [f"--{format_}", f"{structure_org_filename}"]
        if isinstance(offset_file, SinglefileData):
            _filename = offset_file.filename
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
            filename = offset_file.filename
            pattern_files.append(filename)

        calcinfo.retrieve_list = pattern_files

        return calcinfo


class ExtractParser(Parser):

    def parse(self, **kwargs):

        _cwd = ""
        if "cwd" in self.node.inputs:
            _cwd = self.node.inputs.cwd.value
        cwd = _cwd

        prefix = self.node.inputs.prefix.value

        if len(cwd) > 0:
            os.makedirs(cwd, exist_ok=True)

        try:
            output_folder = self.retrieved
        except Exception:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        _, _ = save_output_folder_files(output_folder, cwd, prefix)

        filename = self.node.get_option('output_filename')
        if filename in output_folder.list_object_names():
            _content = output_folder.get_object_content(filename).splitlines()
        else:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        self.out('dfset', List(list=_content))
