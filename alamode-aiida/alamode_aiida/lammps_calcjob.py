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
import io
import math
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from ase.io.espresso import read_espresso_in
from aiida.orm import Str, Int
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory

import os

FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
Dict = DataFactory("dict")
List = DataFactory('list')


class lammps_CalcJob(CalcJob):
    """lammps interface

    potential_filenames is assumed to be in the directory, cwd.

    XFSET_filename is retrived from the remote host and placed as XFSET_alt_filename.
    """
    _WITHMPI = True
    _XFSET = "XFSET"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("input_filename", valid_type=Str)
        spec.input("data_filename", valid_type=Str)
        spec.input("potential_files", valid_type=(List, FolderData))
        spec.input("XFSET_filename", valid_type=Str,
                   default=lambda: Str(cls._XFSET))
        spec.input("XFSET_alt_filename", valid_type=Str,
                   default=lambda: Str(cls._XFSET))

        spec.input("cwd", valid_type=Str)

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.lammps'
        spec.inputs['metadata']['options']['input_filename'].default = 'file.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'file.out'
        spec.input('metadata.options.withmpi',
                   valid_type=bool, default=cls._WITHMPI)
        spec.inputs['metadata']['options']['resources'].default = {'num_machines': 1,
                                                                   'num_mpiprocs_per_machine': 1}
        spec.output('result', valid_type=Dict)
        spec.output('output_file', valid_type=SinglefileData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:

        # lammps input file
        input_file_path = os.path.join(self.inputs.cwd.value,
                                       self.inputs.input_filename.value)

        folder.insert_path(
            input_file_path, dest_name=self.options.input_filename)

        # data file
        data_file_path = os.path.join(self.inputs.cwd.value,
                                      self.inputs.data_filename.value)

        folder.insert_path(
            data_file_path, dest_name=self.inputs.data_filename.value)

        # potential files
        if isinstance(self.inputs.potential_files,List):
            potentials = self.inputs.potential_files.get_list()
            for _target_path in potentials:
                _, _potential_filename = os.path.split(_target_path)
                folder.insert_path(_target_path, dest_name=_potential_filename)
        elif isinstance(self.inputs.potential_files,FolderData):
            potentials = self.inputs.potential_files.list_object_names()
            for _potential_filename in potentials:
                with folder.open(_potential_filename, 
                            'w', encoding='utf8') as handle:
                    handle.write(self.inputs.potential_files.get_object_content(_potential_filename))

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = ["-in", self.options.input_filename]
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = self.options.withmpi

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.input_filename,
                                  self.options.output_filename,
                                  self.inputs.XFSET_filename.value]

        return calcinfo


def _parse_lammps_out_2016(data, result):
    """parse lammps build 9 Nov 2016"""
    data_iter = iter(data)
    result = {}
    while True:
        try:
            line = next(data_iter)
        except StopIteration:
            is_finished_ok = False
            break
        if line.startswith("LAMMPS"):
            s = line.replace("LAMMPS", "").replace(
                "(", "").replace(")", "").strip()
            result["lammps_build"] = s
        if line.startswith("Ave neighs/atom ="):
            s = line.split("=")
            result["ave_neighs_per_atom"] = int(s[1].strip())
        elif line.startswith("Total wall time:"):
            s = line.replace("Total wall time:", "")
            result["total_wall_time"] = s.strip()
            is_finished_ok = True
            break

    return result


def _parse_lammps_out(filename: str = None, handle: io.TextIOWrapper = None):
    """parse lammps output file.

    is_finished_ok = True if "Total wall time:" is found

    Args:
        filename (str, optional): an output filename of pw.x. Defaults to None.
        handle (io.TextIOWrapper, optional): file hander. Defaults to None.

    Returns:
        dict: output parameters.
    """
    if handle is None and filename is not None:
        with open(filename) as f:
            data = f.read().splitlines()
    elif handle is not None and filename is None:
        data = handle.read().splitlines()

    version = None
    data_iter = iter(data)
    while True:
        try:
            line = next(data_iter)
        except StopIteration:
            is_finished_ok = False
            break
        if line.startswith("LAMMPS"):
            version = line.replace("LAMMPS", "").replace(
                "(", "").replace(")", "").strip()
            break

    if version is None:
        raise ValueError("No lammps version is found.")

    result = {}

    if version == '9 Nov 2016':
        result = _parse_lammps_out_2016(data, result)
    else:
        raise ValueError(f"unknown lammps version = {version}")

    return result


class lammps_ParseJob(Parser):

    def parse(self, **kwargs):
        try:
            output_folder = self.retrieved
        except:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        try:
            with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                result = _parse_lammps_out(handle=handle)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        except ValueError:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        self.out('result', Dict(dict=result))

        cwd = self.node.inputs.cwd.value

        _filename = self.node.get_option('output_filename')
        _content = output_folder.get_object_content(_filename)
        target_path = os.path.join(cwd, _filename)
        with open(target_path, "w") as f:
            f.write(_content)

        xfset_filename = self.node.inputs.XFSET_filename.value
        _content = output_folder.get_object_content(xfset_filename)
        xfset_alt_filename = self.node.inputs.XFSET_alt_filename.value
        target_path = os.path.join(cwd, xfset_alt_filename)
        with open(target_path, "w") as f:
            f.write(_content)
        singlefile = SinglefileData(target_path)

        self.out('output_file', singlefile)
