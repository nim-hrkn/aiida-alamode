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
from ..io.aiida_support import save_output_folder_files
import numpy as np
from aiida.orm import Str, Dict
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory

from aiida.common.exceptions import InputValidationError

from ..alamode.analyze_phonons import print_thermal_conductivity_with_boundary
from ..alamode.analyze_phonons import print_temperature_dep_lifetime
from ..alamode.analyze_phonons import print_lifetime_at_given_temperature
from ..alamode.analyze_phonons import print_cumulative_thermal_conductivity

from ..common.base import alamodeBaseCalcjob
from ..io import parse_analyze_phonons_kappa_boundary, parse_analyze_phonons_tau_at_temperature, parse_analyze_phonons_cumulative


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


class analyze_phonons_CalcJob(alamodeBaseCalcjob):
    """_summary_

    analyze_anonons.

    calc can be 'kappa_boundary', 'tau' and 'cumulative'.

    If 'cwd' is given. The retrieved files will be saved in the directory specified by 'cwd'.

    kappa_boundary_file, tau_file and cumulative_file are generated even if 'cwd' isn't given.
    """
    _PARAM_DEFAULT = {}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str, required=False)
        # spec.input("norder", valid_type=Int)
        spec.input("prefix", valid_type=Str)
        spec.input("calc", valid_type=Str)
        # spec.input("size", valid_type=Float)
        # spec.input("temp", valid_type=Float)
        spec.input("file_result", valid_type=(Str, SinglefileData))
        spec.input("param", valid_type=Dict,
                   default=lambda: Dict(dict=cls._PARAM_DEFAULT))
        spec.input("output_filename", valid_type=Str,
                   default=lambda: Str(_OUTPUT_FILENAME_DEFAULT))

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.analyze_phonons'
        spec.inputs['metadata']['options']['input_filename'].default = 'analyze_phonons.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'analyze_phonons.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)
        spec.output('kappa_boundary_file', valid_type=SinglefileData)
        # spec.output('kappa', valid_type=ArrayData)
        spec.output('tau_file', valid_type=SinglefileData)
        # spec.output('tau', valid_type=ArrayData)
        spec.output('cumulative_file', valid_type=SinglefileData)
        # spec.output('cumulative', valid_type=ArrayData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        calc_value = self.inputs.calc.value

        if calc_value == "kappa_boundary":
            result_filename = self.inputs.file_result.filename

            try:
                param = analzePhononOptions(
                    calc_value, **self.inputs.param.get_dict())
            except ValueError as err:
                raise InputValidationError(str(err))

            cmdline = print_thermal_conductivity_with_boundary("", calc_value,
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
            calcinfo.local_copy_list = [(self.inputs.file_result.uuid,
                                         self.inputs.file_result.filename,
                                         self.inputs.file_result.filename)]
            # add files to retrieve list
            retrieve_list = ['_aiidasubmit.sh', self.options.output_filename]
            calcinfo.retrieve_list = retrieve_list

            return calcinfo

        elif calc_value == "tau":
            result_filename = self.inputs.file_result.filename

            try:
                param = analzePhononOptions(
                    calc_value, **self.inputs.param.get_dict())
            except ValueError as err:
                raise InputValidationError(str(err))

            if param.temp is None:
                cmdline = print_temperature_dep_lifetime("", calc_value,
                                                         result_filename,
                                                         param, return_cmd=True)
            else:
                cmdline = print_lifetime_at_given_temperature("", calc_value,
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
            calcinfo.local_copy_list = [(self.inputs.file_result.uuid,
                                         self.inputs.file_result.filename,
                                         self.inputs.file_result.filename)]
            # add files to retrieve list
            retrieve_list = ['_aiidasubmit.sh', self.options.output_filename]
            calcinfo.retrieve_list = retrieve_list

            return calcinfo

        elif calc_value == "cumulative":
            result_filename = self.inputs.file_result.filename

            try:
                param = analzePhononOptions(
                    calc_value, **self.inputs.param.get_dict())
            except ValueError as err:
                raise InputValidationError(str(err))

            cmdline = print_cumulative_thermal_conductivity("", calc_value,
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
            calcinfo.local_copy_list = [(self.inputs.file_result.uuid,
                                         self.inputs.file_result.filename,
                                         self.inputs.file_result.filename)]
            # add files to retrieve list
            retrieve_list = ['_aiidasubmit.sh', self.options.output_filename]
            calcinfo.retrieve_list = retrieve_list

            return calcinfo
        else:
            raise ValueError(f"unknown calc={calc_value}.")


class analyze_phonons_ParseJob(Parser):

    def parse(self, **kwargs):
        calc = self.node.inputs.calc.value
        _cwd = ""
        if "cwd" in self.node.inputs:
            _cwd = self.node.inputs.cwd.value
        cwd = _cwd
        prefix = self.node.inputs.prefix.value
        if calc == "kappa_boundary":
            try:
                output_folder = self.retrieved
            except Exception:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            # retrieve all the files
            _, exit_code = save_output_folder_files(output_folder,
                                                    cwd, prefix)

            filename = self.node.get_option('output_filename')
            if filename not in output_folder.list_object_names():
                return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            with output_folder.open(filename, "rb") as handle:
                self.out('kappa_boundary_file', SinglefileData(handle))

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    try:
                        size, header, values = parse_analyze_phonons_kappa_boundary(
                            handle)
                    except Exception:
                        return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE

            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            if False:
                kappa = ArrayData()
                kappa.set_array('values', np.array(values).astype(float))
                kappa.set_array('columns', np.array(header))
                self.out("kappa", kappa)

            self.out('results', Dict(dict={"size": size}))

        elif calc == "tau":
            try:
                output_folder = self.retrieved
            except Exception:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
            _, exit_code = save_output_folder_files(output_folder,
                                                    cwd, prefix)

            filename = self.node.get_option('output_filename')
            if filename not in output_folder.list_object_names():
                return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            with output_folder.open(filename, "rb") as handle:
                self.out('tau_file', SinglefileData(handle))

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    try:
                        result, header, values = parse_analyze_phonons_tau_at_temperature(handle)
                    except Exception:
                        return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            if False:
                kappa = ArrayData()
                kappa.set_array('values', np.array(values).astype(float))
                kappa.set_array('columns', np.array(header))
                self.out("tau", kappa)

            self.out('results', Dict(dict=result))

        elif calc == "cumulative":
            try:
                output_folder = self.retrieved
            except Exception:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
            _, exit_code = save_output_folder_files(output_folder,
                                                    cwd, prefix)

            filename = self.node.get_option('output_filename')
            if filename not in output_folder.list_object_names():
                return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            with output_folder.open(filename, "rb") as handle:
                self.out('cumulative_file', SinglefileData(handle))

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    try:
                        result, header, values = parse_analyze_phonons_cumulative(handle)
                    except Exception:
                        return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            if False:
                kappa = ArrayData()
                kappa.set_array('values', np.array(values).astype(float))
                kappa.set_array('columns', np.array(header))
                self.out("cumulative", kappa)

            self.out('results', Dict(dict=result))
