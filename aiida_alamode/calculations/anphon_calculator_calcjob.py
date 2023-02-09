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
from sys import prefix
from aiida.orm import Str, Dict, Int, Float
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory

import os


from aiida.common.exceptions import InputValidationError

from ..io.aiida_support import folder_prepare_object, save_output_folder_files
from ..common.base import AlamodeBaseCalculation
from .compute_phonon_props import PhononCalculator

AU2ANG = 0.529177


StructureData = DataFactory('structure')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')
ArrayData = DataFactory('array')


class AnphonCalculatorCalculation(AlamodeBaseCalculation):
    """
    anphon using PhononCalculator.

    mode can be 'phonons'

    for mode=='phonons', phonons_mode can be 'band', 'dos' or 'dos+thermo'.

    """
    _WITHMPI = False
    _NORDER = 1  # dummy
    _PREFIX_DEFAULT = "alamode"
    _MODE = "phonon"
    _PHONONS_MODE = 'dos'
    _PARAM = {}
    _KAPPA_SPEC = 0
    _QMESH_LIST = [20, 20, 20]
    _FCS_FILENAME = 'anphonon_fcs.xml'
    _QSPACING = 0.1

    @ classmethod
    def define(cls, spec):

        super().define(spec)
        spec.input("structure", valid_type=StructureData,
                   help='primitive structure.')
        spec.input("super_structure", valid_type=StructureData,
                   help='super structure.')
        spec.input("cwd", valid_type=Str, required=False,
                   help='directory where results are saved.')
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX_DEFAULT),
                   help='prefix to be add (This is different from ALM.prefix).')
        spec.input('fcsxml', valid_type=(SinglefileData, List), required=False,
                   help='Probably it is input_ANPHON. It is used at norder=2.')
        spec.input('fcs_filename', valid_type=Str, default=lambda: Str(cls._FCS_FILENAME),
                   help='fcs_filename. It is used at norder=2.')
        spec.input('fc', valid_type=ArrayData, required=False,
                   help='force constant.')
        spec.input('mode', valid_type=Str, default=lambda: Str(
            cls._MODE), help='anphon mode')
        spec.input('phonons_mode', valid_type=Str,
                   default=lambda: Str(cls._PHONONS_MODE), help='phonon mode')
        spec.input('qspacing', valid_type=Float,
                   default=lambda: Float(cls._QSPACING))

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.anphon_calculator'
        spec.inputs['metadata']['options']['input_filename'].default = 'phonon.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'phonon.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)
        spec.output('properties', valid_type=ArrayData, required=False,
                    help='DOS, free energies, energies,...')
        spec.output('bands', valid_type=SinglefileData, required=False,
                    help='bands file.')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        mode = self.inputs.mode.value

        if mode == "phonons":

            # copy dfset_filename
            # copy dfset_filename
            if "fcsxml" in self.inputs:
                try:
                    fcsxml_filename = folder_prepare_object(folder, self.inputs.fcsxml,
                                                            filename=self._FCS_FILENAME, actions=(SinglefileData, List))
                except ValueError as err:
                    raise InputValidationError(str(err))
                except TypeError as err:
                    raise InputValidationError(str(err))

            superstructure = self.inputs.super_structure.get_ase()

            if 'fc' in self.inputs:
                from aiida_alamode.io import Fcsxml
                fcsxml = Fcsxml(superstructure.cell, superstructure.get_scaled_positions(),
                                superstructure.numbers)
                fc2 = self.inputs.fc.get_array('fc2')
                fc_indices = self.inputs.fc.get_array('indices')
                fcsxml.set_force_constants(fc2, fc_indices)
                fcsxml_filename = self.inputs.fcs_filename.value
                with folder.open(fcsxml_filename, "wb", encoding='utf8') as f:
                    fcsxml.write(f)

            # make inputfile
            structure = self.inputs.structure.get_ase()
            calculator = PhononCalculator(fname_primitive=structure,
                                          fname_fcs=fcsxml_filename,
                                          qspacing=self.inputs.qspacing.value)
            calculator.prefix = self.inputs.prefix.value

            phonons_mode = self.inputs.phonons_mode.value
            if phonons_mode == "band":

                with folder.open(self.options.input_filename, 'w', encoding='utf8') as handle:
                    calculator.gen_anphon_input_(handle, kpmode=1)

            elif phonons_mode == 'dos+thermo':

                with folder.open(self.options.input_filename, 'w', encoding='utf8') as handle:
                    calculator.gen_anphon_input_(handle, kpmode=2)
            elif phonons_mode == 'thermo':

                with folder.open(self.options.input_filename, 'w', encoding='utf8') as handle:
                    calculator.gen_anphon_input_(handle, kpmode=2)
            else:
                raise InputValidationError(
                    f"unknown phonons_mode={phonons_mode}")

            # code
            codeinfo = CodeInfo()
            codeinfo.code_uuid = self.inputs.code.uuid
            codeinfo.cmdline_params = [self.options.input_filename]
            codeinfo.stdout_name = self.options.output_filename
            codeinfo.withmpi = self.options.withmpi

            calcinfo = CalcInfo()
            calcinfo.codes_info = [codeinfo]

            retrieve_list = ['_aiidasubmit.sh', self.options.input_filename,
                             self.options.output_filename, fcsxml_filename]
            if phonons_mode == "band":
                for ext in ["bands"]:
                    filename = f"{calculator.prefix}.{ext}"
                    retrieve_list.append(filename)
            elif phonons_mode == 'dos+thermo':
                for ext in ["dos", "thermo"]:
                    filename = f"{calculator.prefix}.{ext}"
                    retrieve_list.append(filename)
            elif phonons_mode == 'thermo':
                for ext in ["thermo"]:
                    filename = f"{calculator.prefix}.{ext}"
                    retrieve_list.append(filename)

            calcinfo.retrieve_list = retrieve_list
            return calcinfo


class AnphonCalculatorParser(Parser):

    def parse(self, **kwargs):
        mode = self.node.inputs.mode.value
        alm_prefix_node = self.node.inputs.prefix

        _cwd = ""
        if "cwd" in self.node.inputs:
            _cwd = self.node.inputs.cwd.value
        cwd = _cwd
        if len(cwd) > 0:
            os.makedirs(cwd, exist_ok=True)

        if mode == "phonons":
            try:
                output_folder = self.retrieved
            except Exception:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
            _, exit_code = save_output_folder_files(output_folder,
                                                    cwd, alm_prefix_node)

            phonons_mode = self.node.inputs.phonons_mode.value
            array = ArrayData()

            phonon_mode_list = phonons_mode.split("+")
            result = {}
            finished_job = []
            try:
                if 'band' in phonon_mode_list:
                    _, exit_code = save_output_folder_files(output_folder,
                                                            cwd, "band")
                    finished_job.append("band")
                    result.update({"finished": finished_job})
                    band_filename = self.node.inputs.prefix.value+".bands"
                    if band_filename in output_folder.list_object_names():
                        with output_folder.open(band_filename, "rb") as handle:
                            self.out("bands", SinglefileData(handle))

                if 'dos' in phonon_mode_list:
                    _, exit_code = save_output_folder_files(output_folder,
                                                            cwd, "dos")
                    calculator = PhononCalculator(fname_primitive=self.node.inputs.structure.get_ase(),
                                                  fname_fcs=self.node.inputs.fcs_filename.value,
                                                  qspacing=self.node.inputs.qspacing.value)
                    calculator.prefix = self.node.inputs.prefix.value
                    filename = self.node.inputs.prefix.value+".dos"
                    with output_folder.open(filename, 'r') as handle:
                        try:
                            calculator.parse_dos_(handle)
                        except ValueError:
                            return self.exit_codes.ERROR_INVALID_OUTPUT
                    array.set_array('dos', calculator.dos)
                    finished_job.append("dos")
                    result.update({"finished": finished_job})

                if 'thermo' in phonon_mode_list:
                    _, exit_code = save_output_folder_files(output_folder,
                                                            cwd, "thermo")
                    calculator = PhononCalculator(fname_primitive=self.node.inputs.structure.get_ase(),
                                                  fname_fcs=self.node.inputs.fcs_filename.value,
                                                  qspacing=self.node.inputs.qspacing.value)
                    calculator.prefix = self.node.inputs.prefix.value
                    filename = self.node.inputs.prefix.value+".thermo"
                    with output_folder.open(filename, 'r') as handle:
                        try:
                            calculator.parse_thermo_(handle)
                        except ValueError:
                            return self.exit_codes.ERROR_INVALID_OUTPUT
                    array.set_array('temperatures', calculator.temperatures)
                    array.set_array('heat_capacity', calculator.heat_capacity)
                    array.set_array('entropy', calculator.entropy)
                    array.set_array('internal_energy', calculator.internal_energy)
                    array.set_array('free_energy', calculator.free_energy)

                    with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                        calculator.parse_frequencies_(handle)
                    array.set_array('frequencies', calculator.frequcncies)

                    omega_lowest = calculator.get_lowest_frequency()
                    calculator._detect_imaginary_branches()

                    finished_job.append("thermo")
                    result.update({"finished": finished_job,
                                   "imaginary_ratio": calculator.get_imaginary_ratio(),
                                   'omega_lowest': omega_lowest})

                if 'dos' in phonon_mode_list or 'thermo' in phonon_mode_list:
                    self.out('properties', array)

            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            self.out('results', Dict(dict=result))
