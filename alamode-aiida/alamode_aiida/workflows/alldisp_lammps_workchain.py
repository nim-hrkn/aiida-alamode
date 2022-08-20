from aiida.orm import Code
from aiida.orm import Str, Dict
from aiida.engine import calcfunction, WorkChain, append_
from aiida.plugins import DataFactory
from itertools import cycle
import os

from ..io.misc import zerofillStr

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
LammpsPotential = DataFactory('lammps.potential')
TrajectoryData = DataFactory('array.trajectory')


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
        raise ValueError(
            f"unknown type ={type(codelist)} in _filelist_to_code_list")
    codelist = []
    for _ in filelist:
        _code_string = next(code_pool)

        codelist.append(_code_string)
    return Str(codelist[index.value])


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
    result = [v for v in kwargs.values()]
    return List(list=result)


@calcfunction
def _pack_key_dict(**kwargs):
    format = "LAMMPS"
    result = [result
              for label, result in kwargs.items()]
    result = {format: result}
    return Dict(dict=result)


@calcfunction
def _pack_value(**kwargs):
    result = [result
              for label, result in kwargs.items()]
    return List(list=result)


@calcfunction
def _pack_Dict(**kwargs):
    result = [result.attributes
              for label, result in kwargs.items()]
    return List(list=result)


@calcfunction
def _pack_to_FolderData(**kwargs):
    folderdata = FolderData()
    for filepath in kwargs.values():
        _, filename = os.path.split(filepath)
        folderdata.put_object_from_file(filepath, path=filename)
    return folderdata


@calcfunction
def _pack_retrieved_to_FolderData(calculations, object_name, cwd, prefix,
                                  filename_template):
    cwd = cwd.value
    prefix = prefix.value
    object_name = object_name.value
    filename_template = filename_template.value

    logfolder = FolderData()
    for _i, w in enumerate(calculations):
        counter = zerofillStr.str(_i+1)  # counter for output file
        folder = w.get_outgoing().get_node_by_label('retrieved')
        content = folder.get_object_content(object_name)
        filename = filename_template.replace(
            "{prefix}", prefix).replace("{counter}", counter)
        filepath = os.path.join(cwd, filename)
        with open(filepath, "w") as f:
            f.write(content)
        logfolder.put_object_from_file(filepath, path=filename)
    return logfolder


def _change_lammps_section(cwd,
                           lammps_input_file_template,
                           replace_dic,
                           lammps_input_filename):
    change = replace_dic.attributes
    lines = lammps_input_file_template.get_content().splitlines()

    for key, value in change.items():
        line2 = []
        for line in lines:
            s = line.split()
            flag = True
            if len(s) > 0:
                if s[0] == key:
                    line2.append(" ".join([s[0], value]))
                    flag = False
            if flag:
                line2.append(line)
        lines = line2
    lammps_new_input_filepath = os.path.join(
        cwd.value, lammps_input_filename.value)
    lammps_new_input_filepath = os.path.abspath(lammps_new_input_filepath)
    with open(lammps_new_input_filepath, "w") as f:
        f.write("\n".join(lines))


class force_simulator_lammps_WorkChain(WorkChain):
    """parallen execution of lammps.force
    """
    _CWD = ""
    _WAIT_SEC = 2
    _WORKCHAIN_KEY_FORMAT = "workchain_{}"
    _PREFIX = "disp"
    _WITHMPI = False
    _RESOURCE = {'withmpi': False,
                 'resources': {'num_machines': 1,
                               'num_mpiprocs_per_machine': 1}}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code_string", valid_type=Str, help='label of your \'lammps.force\' code')
        spec.input("structures", valid_type=TrajectoryData, help='dispalced structures')
        spec.input('potential', valid_type=LammpsPotential, help='lammps potential')
        spec.input('parameters', valid_type=Dict, help='additional parameters to pass \'lammps.force\'')

        spec.input("cwd", valid_type=Str,  default=lambda: Str(cls._CWD), help='directory where results are saved.')
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX), help='string added to filenames')

        spec.input('options',
                   valid_type=Dict, default=lambda: Dict(dict=cls._RESOURCE), help='metadata.options')

        spec.outline(
            cls.submit_workchains,
            cls.inspect_workchains
        )
        spec.output("results", valid_type=List)
        spec.output("forces", valid_type=List, help='resulting forces')
        spec.output("displacement_and_forces", valid_type=Dict, help='displacement and forces')
        # spec.output("output_folder", valid_type=FolderData)
        # spec.output("dfset_folder", valid_type=FolderData)

    def submit_workchains(self):
        structures = self.inputs.structures
        potential = self.inputs.potential
        code = self.inputs.code_string.value
        metadata_options = self.inputs.options.get_dict()
        parameters = self.inputs.parameters

        num_disp = len(structures.get_stepids())

        code_lammps_force = Code.get_from_string(code)

        for _i in range(num_disp):

            structure = structures.get_step_structure(_i)

            builder = code_lammps_force.get_builder()
            # builder.metadata.options = meta_options
            builder.structure = structure
            builder.potential = potential
            builder.parameters = parameters
            builder.metadata.options = metadata_options

            future = self.submit(builder)  # or self.submit

            # self.to_context(**{key: future})
            self.to_context(simulator=append_(future))

    def inspect_workchains(self):
        cwd = self.inputs.cwd.value
        if len(cwd) > 0:
            os.makedirs(cwd, exist_ok=True)

        prefix = self.inputs.prefix.value

        for w in self.ctx.simulator:
            assert w.is_finished_ok

        calculations = self.ctx.simulator

        num_disp = len(calculations)

        results = {}
        for _i, w in enumerate(calculations):
            results[f"label{_i}"] = w.get_outgoing(
            ).get_node_by_label('results').get_dict()
        self.out('results', _pack_value(**results))
        results = {}
        for _i, w in enumerate(calculations):
            array = w.get_outgoing().get_node_by_label('arrays')
            results[f"label{_i}"] = array.get_array('forces')
        self.out('forces', _pack_value(**results))

        zerofillstr = zerofillStr(num_disp)

        calculations = List(list=calculations)

        object_name = 'trajectory.lammpstrj'
        filename_template = 'XFSET_{prefix}{counter}'
        results = {}
        for _i, w in enumerate(calculations):
            counter = zerofillstr.str(_i+1)  # counter for output file
            folder = w.get_outgoing().get_node_by_label('retrieved')
            content = folder.get_object_content(object_name)
            results[counter] = content
        self.out('displacement_and_forces', _pack_key_dict(**results))

        if len(cwd) > 0:
            object_name = 'log.lammps'
            filename_template = 'lammps_{prefix}{counter}.out'
            results = {}
            for _i, w in enumerate(calculations):
                counter = zerofillstr.str(_i+1)  # counter for output file
                folder = w.get_outgoing().get_node_by_label('retrieved')
                content = folder.get_object_content(object_name)
                filename = filename_template.replace(
                    "{prefix}", prefix).replace("{counter}", counter)
                filepath = os.path.join(cwd, filename)
                with open(filepath, "w") as f:
                    f.write(content)
                results[f'output{counter}'] = filepath
            # self.out('output_folder', _pack_to_FolderData( **results))

        if len(cwd) > 0:
            object_name = 'trajectory.lammpstrj'
            filename_template = 'XFSET_{prefix}{counter}'
            results = {}
            for _i, w in enumerate(calculations):
                counter = zerofillstr.str(_i+1)  # counter for output file
                folder = w.get_outgoing().get_node_by_label('retrieved')
                content = folder.get_object_content(object_name)
                filename = filename_template.replace(
                    "{prefix}", prefix).replace("{counter}", counter)
                filepath = os.path.join(cwd, filename)
                with open(filepath, "w") as f:
                    f.write(content)
                results[f'dfset{counter}'] = filepath
            # self.out('dfset_folder', _pack_to_FolderData( **results))
