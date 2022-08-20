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
from aiida.orm import Str, Dict, Int
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import calcfunction
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory

import os
from fnmatch import fnmatch

from ..common.base import alamodeBaseCalcjob

from ..io.aiida_support import folder_prepare_object, save_output_folder_files, file_type_conversion

from ..io.alm_input import make_alm_in, atoms_to_alm_in
from ..io.displacement import lines_to_displacementpattern


StructureData = DataFactory('structure')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')
ArrayData = DataFactory('array')


def _parse_alm_suggest_output(filename=None, handle=None):
    """alm suggest parser

    Args:
        filename (str, optional): flename to parse. Defaults to None.
        handle (_type_, optional): file hanlder. Defaults to None.

    Returns:
        Tuples containing
        dict: parsed dictionary.
        int: Error code
    """
    if filename is not None and handle is None:
        with open(filename) as f:
            data = f.read().splitlines()
    elif filename is None and handle is not None:
        data = handle.read().splitlines()

    num_disp = {}
    num_free_fcs = {}
    disp_pattern_filenames = {}
    data_iter = iter(data)
    found_job_finished = False
    while True:
        line = next(data_iter)

        if "Number of free" in line:
            s = line.split()
            num_free_fcs[s[3].strip()] = int(s[-1].strip())
        elif "Number of disp. patterns for" in line:
            s = line.split(":")
            s1 = s[0].split()[-1].strip()
            s2 = s[1].strip()
            num_disp[s1] = s2
        elif "Suggested displacement patterns" in line:
            while True:
                line = next(data_iter).strip()
                if len(line) == 0:
                    break
                s = line.split(":")
                disp_pattern_filenames[s[0].strip()] = s[-1].strip()
        elif "Job finished" in line:
            found_job_finished = True
            break

    if len(num_free_fcs.keys()) == 0:
        raise IOError("'num_free_fcs' could not be found")
    if len(num_disp.keys()) == 0:
        raise IOError("'num_disp' not found")
    if len(disp_pattern_filenames.keys()) == 0:
        raise IOError("'disp_pattern_filenames' could not be found")
    if not found_job_finished:
        raise IOError("'Job finished' could not be found")

    result = {"num_free_fcs,": num_free_fcs, "num_disp": num_disp,
              "disp_pattern_filenames": disp_pattern_filenames}

    return result


def _alm_pattern_file(prefix, norder):
    if norder == 1:
        term = "HARMONIC"
    elif norder == 2:
        term = "ANHARM3"
    else:
        raise ValueError(f"uknown order={norder}")
    return f"{prefix}.pattern_{term}"


def _alm_suggest_retrieve_pattern_file_as_Dict(retrieved: Folder, prefix: Str,
                                               ) -> dict:
    prefix_value = prefix.value
    result_dic = {}
    for filename in retrieved.list_object_names():
        if fnmatch(filename, f"{prefix_value}.pattern_*"):
            key = filename
            result_dic[key] = retrieved.get_object_content(
                filename).splitlines()
    return result_dic


@calcfunction
def _alm_suggest_retrieve_pattern_file(result_dic: Dict,
                                       cwd: Str) -> FolderData:
    """
    put files in result_dic in the cwd directory.

    cwd.value must not be "".
    result_dic is the result of _alm_suggest_retrieve_pattern_file_as_Dict.

    Args:
        result_dic (Dict): pattern file as Dict.
        cwd (Str): the directory to put files.

    Returns:
        FolderData: pattern files.
    """
    cwd_value = cwd.value
    if len(cwd.value) > 0:
        result_dic = result_dic.get_dict()
        folderdata = FolderData()
        for filename, _content in result_dic.items():
            target_path = os.path.join(cwd_value, filename)
            with open(target_path, "w") as f:
                f.write(_content)
            folderdata.put_object_from_file(target_path, path=filename)
        return folderdata
    else:
        return None


def _parse_alm_opt(handle):
    data = handle.read().splitlines()

    data_iter = iter(data)
    constraint = {}
    optimization = {}
    outputfiles = {}
    warning_messages = []
    job_finished = False
    while True:
        line = next(data_iter)

        if "Number of constraints [T-inv, R-inv (self), R-inv (cross)]" in line:
            line = next(data_iter)
            s = line.split()
            constraint["constraint_T-inv"] = int(s[1])
            constraint["constraint_R-inv_self"] = int(s[2])
            constraint["constraint_R-inv_cross"] = int(s[3])
        elif "Number of inequivalent constraints" in line:
            line = next(data_iter)
            s = line.split()
            constraint["inqeuv_constraint_self"] = int(s[1])
            constraint["inqeuv_constraint_cross"] = int(s[2])
        elif "Number of free HARMONIC FCs" in line:
            s = line.split(":")
            constraint["num_free_HARMINC_FCs"] = int(s[-1])

        elif "WARNING" in line:
            s = line.split(":")
            warning_messages.append(s[1].strip())

        elif "LMODEL" in line:
            s = line.split("=")
            optimization["LMODEL"] = s[-1].strip()
        elif "Total Number of Parameters" in line:
            s = line.split(":")
            optimization["num_param"] = int(s[-1])
        elif "Total Number of Free Parameters" in line:
            s = line.split(":")
            optimization["num_free_param"] = int(s[-1])
        elif "Residual sum of squares for the solution" in line:
            s = line.split(":")
            optimization["RSS"] = float(s[-1])
        elif "Fitting error" in line:
            s = line.split(":")
            optimization["fitting_error"] = float(s[-1])
        elif "RANK of the matrix =" in line:
            s = line.split("=")
            optimization["rank_of_matrix"] = int(s[-1])

        elif "Force constants in a human-readable format" in line:
            s = line.split(":")
            outputfiles["force_constants"] = s[-1].strip()
        elif "Input data for the phonon code ANPHON" in line:
            s = line.split(":")
            outputfiles["input_ANPHON"] = s[-1].strip()
        elif 'Job finished' in line:
            job_finished = True
            break

    if len(constraint.keys()) == 0:
        raise IOError("'constraint' could not be found")
    if len(optimization.keys()) == 0:
        raise IOError("'optimization' not found")
    if len(outputfiles.keys()) == 0:
        raise IOError("'outputfiles' could not be found")
    if not job_finished:
        raise IOError("'Job finished' could not be found")

    results = {"constraint": constraint, "optimization": optimization,
               "outputfiles": outputfiles}
    if len(warning_messages) > 0:
        results["warning"] = warning_messages
    return results


class almBaseCalcJob(alamodeBaseCalcjob):
    """
    write files to the cwd directory if cwd is not "".

    If fc2xml_file is SinglefileData, first place it in cwd directory with the same name as SinglefileData.
    """
    _WITHMPI = True

    _PREFIX_DEFAULT = "disp"
    _CUTOFF_DEFAULT = {"*-*": [None]}
    _PARAM_DEFAULT = {}
    _CWD = ""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData,
                   help='structure of cyrstal.')

        spec.input("cwd", valid_type=Str, default=lambda: Str(
            cls._CWD), help='directory where results are saved.')
        spec.input("norder", valid_type=Int, help='1 (harmonic) or 2 (cubic)')
        spec.input("prefix", valid_type=Str,
                   default=lambda: Str(cls._PREFIX_DEFAULT), help='string added to the filename.')
        spec.input('cutoff', valid_type=Dict,
                   default=lambda: Dict(dict=cls._CUTOFF_DEFAULT), help='distance cutoff')

        spec.input('param', valid_type=Dict,
                   default=lambda: Dict(dict=cls._PARAM_DEFAULT), help='optional parameters')

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.alm'
        spec.inputs['metadata']['options']['input_filename'].default = 'alm.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'alm.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)


class almSuggestCalcJob(almBaseCalcJob):
    """alm mode="suggest"

    pattern files are saved as f"{prefix}.pattern_*".

    default input filename: alm_suggest.in
    default output filename: alm_suggest.out

    """
    _WITHMPI = True
    _DFSET_FILE = ""
    _FC2XML_FILE = ""
    _MODE = "suggest"
    _PARAM_DEFAULT = {}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(almBaseCalcJob)
        spec.input('mode', valid_type=Str, default=lambda: Str(cls._MODE),
                   help='mode of alm=\'suggest\'')

        spec.inputs['metadata']['options']['input_filename'].default = 'alm_suggest.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'alm_suggest.out'
        spec.expose_outputs(almBaseCalcJob)
        spec.output('pattern', valid_type=List, help='pattern of displacement')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:

        prefix_value = self.inputs.prefix.value
        norder = self.inputs.norder.value

        pattern_files = []
        for iorder in range(1, norder+1):
            pattern_file = _alm_pattern_file(prefix_value, iorder)
            pattern_files.append(pattern_file)

        # make inputfile
        structure = self.inputs.structure.get_ase()
        cutoff_value = self.inputs.cutoff.get_dict()

        alm_param = atoms_to_alm_in(
            "suggest", structure, prefix=prefix_value, norder=norder,
            cutoff=cutoff_value, dic=self.inputs.param.get_dict())

        with folder.open(self.options.input_filename, 'w', encoding='utf8') as handle:
            make_alm_in(alm_param, handle=handle)

        # code
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [self.options.input_filename]
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = self.options.withmpi

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]

        # add files to retrieve list
        pattern_files.extend(
            [self.options.input_filename, self.options.output_filename])
        calcinfo.retrieve_list = pattern_files

        return calcinfo


class almOptCalcJob(almBaseCalcJob):
    """alm mode="opt"

    default input filename: alm_opt.in
    default output filename: alm_opt.out

    """
    _WITHMPI = True
    _MODE = "opt"
    _DFSET_FILENAME = "DFSET"
    _FC2XML_FILE = ""
    _PREFIX_DEFAULT = "alamode"
    _CUTOFF_DEFAULT = {"*-*": [None]}
    _PARAM_DEFAULT = {}
    _input_ANPHON_file = "alm_opt.xml"
    _force_constants = "alm_opt.fcs"
    _FC2XML = []

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(almBaseCalcJob)
        spec.input('mode', valid_type=Str, default=lambda: Str(cls._MODE),
                   help='mode of alm=\'opt\'')
        spec.input('dfset', valid_type=List, help='DFSET')
        spec.input('fc2xml', valid_type=(List, SinglefileData),
                   default=lambda: List(list=cls._FC2XML), help='xml file for the cubic term')

        spec.inputs['metadata']['options']['input_filename'].default = 'alm_opt.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'alm_opt.out'
        spec.expose_outputs(almBaseCalcJob)
        spec.output('input_ANPHON', valid_type=(List, SinglefileData), help='ANPHON file')
        spec.output('force_constants', valid_type=(List, SinglefileData), help='force constants.')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:

        prefix_value = self.inputs.prefix.value
        norder = self.inputs.norder.value

        # copy dfset_filename
        dfset = self.inputs.dfset.get_list()
        DFSETfilename = self._DFSET_FILENAME
        with folder.open(DFSETfilename, "w") as f:
            f.write("\n".join(dfset))

        fc2xml = self.inputs.fc2xml
        if False:
            if isinstance(fc2xml, List):
                fc2xml = fc2xml.get_list()
                if len(fc2xml) > 0:
                    fc2xml_filename = f"{prefix_value}.xml"
                    if len(fc2xml) > 0:
                        with folder.open(fc2xml_filename, "w") as f:
                            f.write("\n".join(fc2xml))
            elif isinstance(fc2xml, SinglefileData):
                fc2xml_filename = fc2xml.list_object_names()[0]
                with folder.open(fc2xml_filename, "w") as f:
                    f.write(fc2xml.get_content())
            else:
                raise ValueError('unknown type for self.inputs.fc2xml, type=',
                                 type(fc2xml))
        else:
            fc2xml_filename = folder_prepare_object(folder, fc2xml,
                                                    filename=f"{prefix_value}.xml",
                                                    actions=[List, SinglefileData])

        # make inputfile
        structure = self.inputs.structure.get_ase()
        param = self.inputs.param.get_dict()
        if "optimize" in param.keys():
            param["optimize"]["DFSET"] = DFSETfilename
        else:
            param["optimize"] = {"DFSET": DFSETfilename}

        if isinstance(fc2xml, List):
            if len(fc2xml) > 0:
                param["optimize"]["FC2XML"] = fc2xml_filename  # only basename?
        elif isinstance(fc2xml, SinglefileData):
            param["optimize"]["FC2XML"] = fc2xml_filename

        alm_param = atoms_to_alm_in("opt", structure,
                                    dic=param, norder=norder,
                                    prefix=prefix_value)
        with folder.open(self.options.input_filename, 'w', encoding='utf8') as handle:
            make_alm_in(alm_param, handle=handle)

        # code
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [self.options.input_filename]
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = self.options.withmpi

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [
            self.options.input_filename, self.options.output_filename]

        for ext in ["fcs", "xml"]:
            filename = f"{prefix_value}.{ext}"
            calcinfo.retrieve_list.append(filename)

        return calcinfo


class alm_ParseJob(Parser):

    def parse(self, **kwargs):
        print("alm_ParseJob start")
        mode = self.node.inputs.mode.value
        cwd = self.node.inputs.cwd.value
        alm_prefix_node = self.node.inputs.prefix

        if len(cwd) > 0:
            # create the directory if it isn't exist.
            os.makedirs(cwd, exist_ok=True)

        if mode == "optimize":
            mode = "opt"

        if mode == "suggest":
            print("mode suggest")
            try:
                output_folder = self.retrieved
            except Exception:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    try:
                        result = _parse_alm_suggest_output(handle=handle)
                    except Exception:
                        _, exit_code = save_output_folder_files(output_folder,
                                                                cwd, alm_prefix_node)
                        raise self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT
            print("result done")
            pattern_file_dict = _alm_suggest_retrieve_pattern_file_as_Dict(output_folder,
                                                                           alm_prefix_node)
            if (pattern_file_dict.keys()) == 0:
                _, exit_code = save_output_folder_files(output_folder,
                                                        cwd, alm_prefix_node)
                return self.exit_codes.ERROR_UNEXPECTED_PARSER_EXCEPTION

            # arrange by result["disp_pattern"]
            patern_lines = []
            for key in ["HARMONIC", "ANHARM3"]:
                if key in result["disp_pattern_filenames"]:
                    filename = result["disp_pattern_filenames"][key]
                    lines = pattern_file_dict[filename]
                    patern_lines.append(lines_to_displacementpattern(lines))
            if len(patern_lines) == 0:
                _, exit_code = save_output_folder_files(output_folder,
                                                        cwd, alm_prefix_node)
                return self.exit_codes.ERROR_OUTPUT_PATTERN_FILES_MISSING

            _, exit_code = save_output_folder_files(output_folder, cwd, alm_prefix_node)

            self.out('results', Dict(dict=result))
            self.out('pattern', List(list=patern_lines))

        elif mode == "opt":
            try:
                output_folder = self.retrieved
            except Exception:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    result = _parse_alm_opt(handle=handle)
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            filelist = [self.node.get_option('input_filename'),
                        self.node.get_option('output_filename')]

            processed_list = []
            for filename in filelist:
                if filename in output_folder.list_object_names():
                    _content = output_folder.get_object_content(filename)
                    target_path = os.path.join(cwd, filename)
                    with open(target_path, "w") as f:
                        f.write(_content)
                    processed_list.append(filename)
                else:
                    if filename == self.node.get_option('input_filename'):
                        _, exit_code = save_output_folder_files(output_folder,
                                                                cwd, alm_prefix_node)
                        return self.exit_codes.ERROR_OUTPUT_STDIN_MISSING
                    elif filename == self.node.get_option('output_filename'):
                        _, exit_code = save_output_folder_files(output_folder,
                                                                cwd, alm_prefix_node)
                        return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

            conversion_table, _ = save_output_folder_files(output_folder, cwd,
                                                           alm_prefix_node, except_list=processed_list)

            output_type = SinglefileData

            key = "input_ANPHON"
            filename = result["outputfiles"][key]
            filename = conversion_table[filename]

            output_data, exit_msg = file_type_conversion(cwd, filename, output_type)
            if exit_msg == 'NOFILE':
                _, exit_code = save_output_folder_files(output_folder,
                                                        cwd, alm_prefix_node)
                return self.exit_codes.ERROR_OUTPUT_XML_MISSING
            self.out(key, output_data)

            key = "force_constants"
            filename = result["outputfiles"][key]
            filename = conversion_table[filename]

            output_data, exit_msg = file_type_conversion(cwd, filename, output_type)
            if exit_msg == 'NOFILE':
                _, exit_code = save_output_folder_files(output_folder,
                                                        cwd, alm_prefix_node)
                return self.exit_codes.ERROR_OUTPUT_FCS_MISSING
            self.out(key, output_data)

            self.out('results', Dict(dict=result))
