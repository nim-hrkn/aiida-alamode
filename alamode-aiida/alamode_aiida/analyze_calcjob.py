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
import re
import numpy as np
from aiida.orm import Str, Dict, Int, Float
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory


from alamode.analyze_phonons import print_thermal_conductivity_with_boundary
from alamode.analyze_phonons import print_temperature_dep_lifetime
from alamode.analyze_phonons import print_lifetime_at_given_temperature
from alamode.analyze_phonons import print_cumulative_thermal_conductivity


import os


SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory("array")


class analzePhononOptions(object):
    def __init__(self, calc: str, **kwargs):
        self.options = {'temp': None, 'mode': None, 'kpoint': None,
                        'calc': calc, 'isotope': None, 'average_gamma': True,
                        'size': None, 'length': None, 'direction': None}

        for label, value in kwargs.items():
            self.options[label] = value

        if calc == "kappa_boundary":
            if self.options["size"] is None:
                raise ValueError(f"size is necessary for calc={calc}")

        elif calc == "temp":
            if self.options["temp"] is None:
                raise ValueError(f"temp is necessary for calc={calc}")

        elif calc == "cumulative ":
            if self.options["temp"] is None:
                raise ValueError(f"temp is necessary for calc={calc}")
            if self.options["length"] is None:
                raise ValueError(f"length is necessary for calc={calc}")

    def __getattr__(self, name):
        return self.__dict__["options"][name]


_OUTPUT_FILENAME_DEFAULT = "None"


class analyze_phonon_CalcJob(CalcJob):
    _PARAM_DEFAULT = {}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str)
        #spec.input("norder", valid_type=Int)
        spec.input("prefix", valid_type=Str)
        spec.input("calc", valid_type=Str)
        #spec.input("size", valid_type=Float)
        #spec.input("temp", valid_type=Float)
        spec.input("file_result", valid_type=(Str, SinglefileData))
        spec.input("param", valid_type=Dict,
                   default=lambda: Dict(dict=cls._PARAM_DEFAULT))
        spec.input("output_filename", valid_type=Str,
                   default=lambda: Str(_OUTPUT_FILENAME_DEFAULT))

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.analyze_phonon'
        spec.inputs['metadata']['options']['input_filename'].default = 'analyze_phonon.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'analyze_phonon.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('result', valid_type=Dict)
        spec.output('kappa_boundary_file', valid_type=SinglefileData)
        spec.output('kappa', valid_type=ArrayData)
        spec.output('tau_file', valid_type=SinglefileData)
        spec.output('tau', valid_type=ArrayData)
        spec.output('cumulative_file', valid_type=SinglefileData)
        spec.output('cumulative', valid_type=ArrayData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        calc_value = self.inputs.calc.value
        cwd = self.inputs.cwd.value
        print("cwd", cwd)

        if calc_value == "kappa_boundary":
            result_filename = self.inputs.file_result.list_object_names()[0]
            folder.insert_path(os.path.join(cwd, result_filename),
                               dest_name=result_filename)

            param = analzePhononOptions(
                calc_value, **self.inputs.param.get_dict())
            cmdline = print_thermal_conductivity_with_boundary(None, calc_value,
                                                               result_filename,
                                                               param, return_cmd=True)

            # code
            codeinfo = CodeInfo()
            codeinfo.code_uuid = self.inputs.code.uuid
            codeinfo.cmdline_params = cmdline
            codeinfo.stdout_name = self.options.output_filename
            codeinfo.withmpi = self.options.withmpi

            calcinfo = CalcInfo()
            calcinfo.codes_info = [codeinfo]

            # add files to retrieve list
            retrieve_list = [self.options.output_filename]
            calcinfo.retrieve_list = retrieve_list

            return calcinfo

        elif calc_value == "tau":
            result_filename = self.inputs.file_result.list_object_names()[0]
            folder.insert_path(os.path.join(cwd, result_filename),
                               dest_name=result_filename)

            param = analzePhononOptions(
                calc_value, **self.inputs.param.get_dict())

            if param.temp is None:
                cmdline = print_temperature_dep_lifetime(None, calc_value,
                                                         result_filename,
                                                         param, return_cmd=True)
            else:
                cmdline = print_lifetime_at_given_temperature(None, calc_value,
                                                              result_filename, 
                                                              param, return_cmd=True)

            # code
            codeinfo = CodeInfo()
            codeinfo.code_uuid = self.inputs.code.uuid
            codeinfo.cmdline_params = cmdline
            codeinfo.stdout_name = self.options.output_filename
            codeinfo.withmpi = self.options.withmpi

            calcinfo = CalcInfo()
            calcinfo.codes_info = [codeinfo]

            # add files to retrieve list
            retrieve_list = [self.options.output_filename]
            calcinfo.retrieve_list = retrieve_list

            return calcinfo

        elif calc_value == "cumulative":
            result_filename = self.inputs.file_result.list_object_names()[0]
            folder.insert_path(os.path.join(cwd, result_filename),
                               dest_name=result_filename)

            param = analzePhononOptions(
                calc_value, **self.inputs.param.get_dict())

            print("param",param)
            cmdline = print_cumulative_thermal_conductivity(None, calc_value,
                                                        result_filename,
                                                        param, return_cmd=True)
            print("cmdline", cmdline)
            # code
            codeinfo = CodeInfo()
            codeinfo.code_uuid = self.inputs.code.uuid
            codeinfo.cmdline_params = cmdline
            codeinfo.stdout_name = self.options.output_filename
            codeinfo.withmpi = self.options.withmpi

            calcinfo = CalcInfo()
            calcinfo.codes_info = [codeinfo]

            # add files to retrieve list
            retrieve_list = [self.options.output_filename]
            calcinfo.retrieve_list = retrieve_list

            return calcinfo
        else:
            raise ValueError(f"unknown calc={calc_value}.")

def _parse_analyze_phonon_kappa_boundary(handle):
    data = handle.read().splitlines()

    for line in data:
        if line.startswith("# Size of boundary"):
            s = line.split()
            size = " ".join(s[-2:-1])

    for _i, line in enumerate(data):
        if line.startswith(" "):
            data_start = _i
            break
        lastline = line
    header = lastline

    v = re.split("[,()]+", header[1:])
    v2 = []
    for _x in v:
        _x = _x.strip()
        if len(_x) > 0:
            v2.append(_x)
    header = v2

    varname_unit = header[1]
    del header[1]
    varname_unit = varname_unit.split()

    unit = varname_unit[-1]
    varname = varname_unit[0]

    varlist = []
    for _x in header[1:]:
        varlist.append(f"{varname}_{_x} {unit}")
    header = [header[0]]
    header.extend(varlist)

    values = []
    for line in data[data_start:]:
        line = line.strip()
        s = re.split(" +", line)
        v = list(map(float, s))
        values.append(v)

    return size, header, values


def _parse_analyze_phonon_tau_at_temperature(handler):
    data = handler.read().splitlines()
    print(data)
    for _i, line in enumerate(data):
        if line.startswith("# Phonon lifetime at temperature"):
            s = line.replace(".", "").split()
            temp = " ".join(s[-2:])
            print(temp)
        elif line.startswith("# kpoint range"):
            s = line.replace(".", "").split()
            kpoint_range = s[-2:]
        elif line.startswith("# mode   range"):
            s = line.replace(".", "").split()
            mode_range = s[-2:]
        if line.startswith(" "):
            data_start = _i
            break
        lastline = line

    header = lastline[1:].strip()
    splitted_header = re.split(", *", header)
    unit = splitted_header[-1].split(" ")[-1]
    for _i, _x in enumerate(splitted_header):
        if _x == 'Thermal conductivity par mode (xx':
            varname = _x
            break
    splitted_header = splitted_header[:_i]

    varname = varname.split("(")[0].strip()
    for ax in ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]:
        splitted_header.append(f"{varname} {ax} {unit}")

    values = []
    for _x in data[data_start:]:
        _xx = re.split(" +", _x.strip())
        values.append(list(map(float,_xx)))

    result = {'temp': temp, 'kpoint': kpoint_range, 'mode_range': mode_range}
    return result,  splitted_header,  values

def _parse_analyze_phonon_cumulative(handler):
    data = handler.read().splitlines()
    print(data)
    for _i, line in enumerate(data):
        if line.startswith("# Cumulative thermal conductivity at temperature"):
            s = line.replace(".", "").split()
            temp = " ".join(s[-2:])
            print(temp)
        elif line.startswith("# mode range"):
            s = line.replace(".", "").split()
            mode_range = s[-2:]
        if line.startswith(" "):
            data_start = _i
            break
        lastline = line

    header = lastline[1:].strip()
    splitted_header = re.split(", *", header)
    for _i, _x in enumerate(splitted_header):
        if _x == 'kappa [W/mK] (xx':
            varname = _x
            break
    splitted_header = splitted_header[:_i]
    print(splitted_header)

    varname_unit = varname.split()
    varname = varname_unit[0].strip()
    unit = varname_unit[1].strip()
    for ax in ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]:
        splitted_header.append(f"{varname} {ax} {unit}")

    values = []
    for _x in data[data_start:]:
        _xx = re.split(" +", _x.strip())
        values.append(list(map(float,_xx)))

    result = {'temp': temp,  'mode_range': mode_range}
    return result,  splitted_header,  values

class analyze_phonon_ParseJob(Parser):

    def parse(self, **kwargs):
        print("parse start")
        calc = self.node.inputs.calc.value
        cwd = self.node.inputs.cwd.value
        prefix = self.node.inputs.prefix.value
        if calc == "kappa_boundary":
            try:
                output_folder = self.retrieved
            except:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    size, header, values = _parse_analyze_phonon_kappa_boundary(
                        handle)
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            print("output_folder", output_folder.list_object_names())

            filename = self.node.get_option('output_filename')
            _content = output_folder.get_object_content(filename)
            if self.node.inputs.output_filename.value == _OUTPUT_FILENAME_DEFAULT:
                filename = f"{prefix}_analyze_phonon_{calc}.dat"
            else:
                filename = self.node.inputs.output_filename.value
            target_path = os.path.join(cwd, filename)
            with open(target_path, "w") as f:
                f.write(_content)
            self.out('kappa_boundary_file', SinglefileData(target_path))

            kappa = ArrayData()
            kappa.set_array('values', np.array(values).astype(float))
            kappa.set_array('columns', np.array(header))
            self.out("kappa", kappa)

            self.out('result', Dict(dict={"size": size}))

        elif calc == "tau":
            try:
                output_folder = self.retrieved
            except:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    result, header, values = _parse_analyze_phonon_tau_at_temperature(handle)
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            print("output_folder", output_folder.list_object_names())

            filename = self.node.get_option('output_filename')
            _content = output_folder.get_object_content(filename)
            if self.node.inputs.output_filename.value == _OUTPUT_FILENAME_DEFAULT:
                filename = f"{prefix}_analyze_phonon_{calc}.dat"
            else:
                filename = self.node.inputs.output_filename.value

            target_path = os.path.join(cwd, filename)
            with open(target_path, "w") as f:
                f.write(_content)
            self.out('tau_file', SinglefileData(target_path))

            kappa = ArrayData()
            kappa.set_array('values', np.array(values).astype(float))
            kappa.set_array('columns', np.array(header))
            self.out("tau", kappa)

            self.out('result', Dict(dict=result))

        elif calc == "cumulative":
            try:
                output_folder = self.retrieved
            except:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    result, header, values = _parse_analyze_phonon_cumulative(handle)
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            print("output_folder", output_folder.list_object_names())

            filename = self.node.get_option('output_filename')
            _content = output_folder.get_object_content(filename)
            if self.node.inputs.output_filename.value == _OUTPUT_FILENAME_DEFAULT:
                filename = f"{prefix}_analyze_phonon_{calc}.dat"
            else:
                filename = self.node.inputs.output_filename.value

            target_path = os.path.join(cwd, filename)
            with open(target_path, "w") as f:
                f.write(_content)
            self.out('cumulative_file', SinglefileData(target_path))

            kappa = ArrayData()
            kappa.set_array('values', np.array(values).astype(float))
            kappa.set_array('columns', np.array(header))
            self.out("cumulative", kappa)

            self.out('result', Dict(dict=result))

