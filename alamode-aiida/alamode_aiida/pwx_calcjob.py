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
from aiida.orm import Str, Int
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory


import os

_PWSCF_XML = "pwscf.xml"


FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')


class pwx_CalcJob(CalcJob):
    _WITHMPI = True
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("pwx_input_filename", valid_type=Str)
        #spec.input("pwx_output_filename", valid_type=Str)
        #spec.input("xml_filename", valid_type=Str)
        spec.input("cwd", valid_type=Str)
        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.pwx'
        spec.inputs['metadata']['options']['input_filename'].default = 'file.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'file.out'
        spec.input('metadata.options.withmpi',
                   valid_type=bool, default=cls._WITHMPI)
        spec.inputs['metadata']['options']['resources'].default = {'num_machines': 1,
                                                                   'num_mpiprocs_per_machine': 1}
        spec.output('output_file', valid_type=SinglefileData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        
        _file_path = os.path.join(self.inputs.cwd.value,
                                  self.inputs.pwx_input_filename.value)
        print("_file_path", _file_path, "dest_name", self.options.input_filename)
        folder.insert_path(_file_path, dest_name=self.options.input_filename)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = ["-in", self.options.input_filename]
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = self.options.withmpi

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.input_filename,
                                  self.options.output_filename,
                                  _PWSCF_XML]

        return calcinfo


def _get_d_from_input_filename(filename: str):
    s = filename.replace(".pw.in", "").replace("disp", "")
    return int(s), len(s)


def _make_output_xml_filename(d: int, ndigit: int, ext: str) -> str:
    if ndigit == 1:
        filename = "disp{:01d}.{}".format(d, ext)
    elif ndigit == 2:
        filename = "disp{:02d}.{}".format(d, ext)
    elif ndigit == 3:
        filename = "disp{:03d}.{}".format(d, ext)
    elif ndigit == 4:
        filename = "disp{:04d}.{}".format(d, ext)
    return filename


class pwx_ParseJob(Parser):

    def parse(self, **kwargs):
        try:
            output_folder = self.retrieved
        except:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        cwd = self.node.inputs.cwd.value

        d, ndigit = _get_d_from_input_filename(
            self.node.inputs.pwx_input_filename.value)
        # xml
        filename = _PWSCF_XML
        _contents = output_folder.get_object_content(filename)
        xml_filename = _make_output_xml_filename(d, ndigit, "xml")
        target_path = os.path.join(cwd, xml_filename)
        with open(target_path, "w") as f:
            f.write(_contents)

        # output
        filename = self.node.get_option('output_filename')
        _contents = output_folder.get_object_content(filename)
        pwx_output_filename = _make_output_xml_filename(d, ndigit, "pw.out")
        target_path = os.path.join(cwd, pwx_output_filename)
        with open(target_path, "w") as f:
            f.write(_contents)
        singlefile = SinglefileData(target_path)

        self.out('output_file', singlefile)

