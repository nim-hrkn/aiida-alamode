from aiida.orm import Code
from aiida.orm import Str, Int, Dict
from aiida.engine import calcfunction, WorkChain, ToContext, append_
from aiida.orm import Str, Dict
from aiida.plugins import DataFactory
from itertools import cycle
import os

from aiida.engine import calcfunction, workfunction, submit, run
from aiida.orm import load_code,load_node
from .misc import zerofillStr

# ## parallel submit
# ref.
# https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/workflows/usage.html?highlight=tocontext#submitting-sub-processes
#

# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')


# pack input and code
@calcfunction
def _filelist_to_pwx_input_list(folderdata, index):
    filelist = folderdata.list_object_names()
    return Str(filelist[index.value])


@calcfunction
def _filelist_to_code_list(folderdata, codelist, index):
    filelist = folderdata.list_object_names()
    if isinstance(codelist, Str):
        code_pool = cycle([codelist.value])
    elif isinstance(codelist, List):
        code_pool = cycle(codelist.get_list())
    else:
        raise ValueError(f"unknown type ={type(codelist)} in _filelist_to_code_list")
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
        code_string = _filelist_to_code_list(self.inputs.pwx_input_folder, 
                                             self.inputs.code_string,
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
def _pack_files(**kwargs):
    """kwargs is a list of SinglefileData

    This doesn't work because SinglefileData isn't json-serialized.

    FolderData doesn't work because filename is necessary, either.

    Returns:
        List: a list of all the SinglefileData
    """
    result = [v  for v in kwargs.values()]
    return List(list=result)

@calcfunction
def _pack_result(**kwargs):
    result = [result.attributes
              for label, result in kwargs.items()]
    return List(list=result)

def _change_lammps_section(cwd,
                                   lammps_input_file_template, 
                                   replace_dic,
                                   lammps_input_filename):
    change = replace_dic.attributes
    lines = lammps_input_file_template.get_content().splitlines()

    for key, value in change.items():
        line2=[]
        for line in lines:
            s = line.split()
            flag = True
            if len(s)>0:
                if s[0]==key:
                    line2.append(" ".join([s[0], value]))
                    flag = False
            if flag:
                line2.append(line)
        lines = line2
    lammps_new_input_filepath=os.path.join(cwd.value, lammps_input_filename.value)
    lammps_new_input_filepath =os.path.abspath(lammps_new_input_filepath)
    with open(lammps_new_input_filepath, "w") as f:
        f.write("\n".join(lines))

class alldisp_lammps_WorkChain(WorkChain):
    _WAIT_SEC = 2
    _WORKCHAIN_KEY_FORMAT = "workchain_{}"
    _PREFIX = "disp"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("displace_result", valid_type=Dict)
        spec.input("data_folder", valid_type=FolderData)
        spec.input("input_file_template", valid_type=SinglefileData)
        spec.input("code_string", valid_type=(Str, List))
        spec.input("cwd", valid_type=Str)
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX))
        spec.input('potential_files', valid_type=(List, FolderData))
        spec.outline(
            cls.submit_workchains,
            cls.inspect_workchains
        )
        spec.output("dfset", valid_type=List)
        spec.output("result", valid_type=List)

    def submit_workchains(self):
        cwd_node = self.inputs.cwd
        data_filenames = self.inputs.data_folder
        potential_files = self.inputs.potential_files
        code_list = self.inputs.code_string
        prefix = self.inputs.prefix.value

        num_disp = self.inputs.displace_result["number_of_displacements"]
        zerofillstr = zerofillStr(num_disp)
        for _i in range(num_disp):
            
            code_string = _filelist_to_code_list(
                data_filenames, code_list, Int(_i))
            
            counter = zerofillstr.str(_i+1)
            lammps_input_file_template = self.inputs.input_file_template
            lammps_input_filename = Str(f"lammps_{prefix}{counter}.in")
            data_filename = self.inputs.displace_result["output_filename"].replace("{counter}",
                                                            counter)
            data_filename = Str(data_filename)
            replace_dic = Dict(dict={"read_data": data_filename.value})
            _change_lammps_section(cwd_node,
                                   lammps_input_file_template, 
                                   replace_dic,
                                   lammps_input_filename)
            lammps_output_filename = Str(f"lammps_{prefix}{counter}.out")
            
            XFSET_alt_filename = Str(f"XFSET.{prefix}{counter}")
            
            _code = Code.get_from_string(code_string.value)

            builder = _code.get_builder()
            builder.input_filename = lammps_input_filename
            builder.cwd = cwd_node
            builder.potential_files = potential_files
            builder.data_filename = data_filename
            builder.XFSET_alt_filename = XFSET_alt_filename
            builder.metadata.options.input_filename = lammps_input_filename.value
            builder.metadata.options.output_filename = lammps_output_filename.value

            future = self.submit(builder)
            
            key = self._WORKCHAIN_KEY_FORMAT.format(_i)
            #self.to_context(**{key: future})
            self.to_context(pwx=append_(future))


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

        self.out("dfset", _pack_filename(**inputs))
        # self.out("dfset_folder", _pack_files(**inputs))
        self.out('result', _pack_result(**results))
        