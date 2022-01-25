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
#!/usr/bin/env python
# coding: utf-8


from aiida.orm import Code
from aiida.orm import Str, Int, Dict
from aiida.engine import calcfunction, WorkChain, ToContext, append_
from aiida.orm import Str, Dict
from aiida.plugins import DataFactory
from itertools import cycle

# ## parallel submit
# ref.
# https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/workflows/usage.html?highlight=tocontext#submitting-sub-processes
#


ArrayData = DataFactory('array')
FolderData = DataFactory('folder')
List = DataFactory('list')


# pack input and code
@calcfunction
def _filelist_to_pwx_input_list(folderdata, index):
    filelist = folderdata.list_object_names()
    return Str(filelist[index.value])


@calcfunction
def _filelist_to_code_list(folderdata, codelist, index):
    filelist = folderdata.list_object_names()
    code_pool = cycle(codelist)
    codelist = []
    for _ in filelist:
        _code_string = next(code_pool)

        codelist.append(_code_string)
    return Str(codelist[index.value])


class ScatterInputAndCodeWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("pwx_input_folder", valid_type=FolderData)
        spec.input('code_string', valid_type=List)
        spec.input('index', valid_type=Int)
        spec.outline(cls.rearange_data)
        spec.output('code_string_list', valid_type=Str)
        spec.output('pwx_input_list', valid_type=Str)

    def rearange_data(self):
        filename = _filelist_to_pwx_input_list(
            self.inputs.pwx_input_folder, self.inputs.index)
        code_string = _filelist_to_code_list(self.inputs.pwx_input_folder, self.inputs.code_string,
                                             self.inputs.index)
        self.out("code_string", codestring)
        self.out("pwx_input", filename)

# This subroutine is important.


@calcfunction
def _pack_filename(**kwargs):
    result = [result.attributes['filename']
              for label, result in kwargs.items()]
    return List(list=result)

@calcfunction
def _pack_result(**kwargs):
    result = [result.attributes
              for label, result in kwargs.items()]
    return List(list=result)

class alldisp_pwx_WorkChain(WorkChain):
    _WAIT_SEC = 2
    _WORKCHAIN_KEY_FORMAT = "workchain_{}"
    _NORDER = 1
    _PSEUDOS_DEFAULT = 'SSSP_1.1_efficiency'

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("pwx_input_folder", valid_type=FolderData)
        spec.input("code_string", valid_type=(Str, List))
        spec.input("cwd", valid_type=Str)
        spec.input('pseudos', valid_type=Str,
                   default=lambda: Str(cls._PSEUDOS_DEFAULT))
        spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.outline(
            cls.submit_workchains,
            cls.inspect_workchains
        )
        spec.output("pwx_output", valid_type=List)
        spec.output("result", valid_type=List)

    def submit_workchains(self):
        cwd_node = self.inputs.cwd
        print("cwd_node", cwd_node)
        pwx_input_filenames = self.inputs.pwx_input_folder
        pseudos = self.inputs.pseudos
        code_list = self.inputs.code_string

        for _i, _ in enumerate(pwx_input_filenames.list_object_names()):

            code_string = _filelist_to_code_list(
                pwx_input_filenames, code_list, Int(_i))
            input_filename = _filelist_to_pwx_input_list(
                pwx_input_filenames, Int(_i))
            print("input_filename", input_filename)

            _code = Code.get_from_string(code_string.value)

            builder = _code.get_builder()
            builder.pwx_input_filename = input_filename
            #builder.pwx_output_filename = _dict_key_str_(pwx_dic, output_filename_key)
            #builder.xml_filename = _dict_key_str_(pwx_dic, xml_filename_key)
            builder.cwd = cwd_node
            builder.pseudos = pseudos
            builder.metadata = {
                'options': {
                    'resources': {'tot_num_mpiprocs': 8, 'num_machines': 1}
                }}

            future = self.submit(builder)

            key = self._WORKCHAIN_KEY_FORMAT.format(_i)
            #self.to_context(**{key: future})
            self.to_context(pwx=append_(future))

            #print("future", i, key, future)

    def inspect_workchains(self):
        for w in self.ctx.pwx:
            assert w.is_finished_ok

        calculations = self.ctx.pwx
        inputs = {}
        for _i, w in enumerate(calculations):
            inputs[f"label{_i}"] = w.get_outgoing().get_node_by_label('output_file')

        results = {}
        for _i, w in enumerate(calculations):
            results[f"label{_i}"] = w.get_outgoing().get_node_by_label('result')

        self.out("pwx_output", _pack_filename(**inputs))

        self.out('result', _pack_result(**results))
        
