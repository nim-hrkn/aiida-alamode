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
from aiida.engine import CalcJob, calcfunction
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory

import os
from fnmatch import fnmatch

from sympy import list2numpy

from ..io.alm_input import make_alm_in, atoms_to_alm_in, make_alm_kpoint
from ..io.displacement import lines_to_displacementpattern
from ..io.aiida_support import folder_prepare

AU2ANG = 0.529177


StructureData = DataFactory('structure')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')
ArrayData = DataFactory('array')





class anphon_CalcJob(CalcJob):
    _WITHMPI = False
    _NORDER = 1  # dummy
    _PREFIX_DEFAULT = "alamode"
    _MODE = "phonon"
    _PHONONS_MODE = 'dos'
    _PARAM = {}
    _KAPPA_SPEC = 0
    _QMESH_LIST = [20, 20, 20]
    _FCS_FILENAME = 'anphonon.fcs'

    @ classmethod
    def define(cls, spec):

        super().define(spec)
        spec.input("structure", valid_type=StructureData, help='primitive structure.')
        spec.input("prefix", valid_type=Str,
                   default=lambda: Str(cls._PREFIX_DEFAULT), help='string added to filename.')
        spec.input("cwd", valid_type=Str, help='directory where results are saved.')
        spec.input('norder', valid_type=Int, default=lambda: Int(cls._NORDER), help='1 (harmonic) or 2 (cubic)')
        spec.input('fcsxml', valid_type=(SinglefileData, List), help='Probably it is input_ANPHON. It is used at norder=2.')
        spec.input('mode', valid_type=Str, default=lambda: Str(cls._MODE), help='anphon mode')
        spec.input('phonons_mode', valid_type=Str,
                   default=lambda: Str(cls._PHONONS_MODE),help='phonon mode')
        spec.input('kappa_spec', valid_type=Int,
                   default=lambda: Int(cls._KAPPA_SPEC))
        # spec.input('kparam', valid_type=Dict, default=lambda: Dict(dict=cls._PARAM))
        spec.input('qmesh', valid_type=List,
                   default=lambda: List(list=cls._QMESH_LIST), help='phonon k-mesh')
        spec.input('param', valid_type=Dict,
                   default=lambda: Dict(dict=cls._PARAM), help='additional parameters')

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.anphon'
        spec.inputs['metadata']['options']['input_filename'].default = f'anphon.in'
        spec.inputs['metadata']['options']['output_filename'].default = f'anphon.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)
        spec.output('phband_file', valid_type=SinglefileData)
        spec.output('phdos_file', valid_type=SinglefileData)
        spec.output('thermo_file', valid_type=SinglefileData)
        spec.output('result_file', valid_type=SinglefileData)
        spec.output('kl_file', valid_type=SinglefileData)
        spec.output('kl', valid_type=ArrayData)
        spec.output('kl_spec_file', valid_type=SinglefileData)
        spec.output('kl_spec', valid_type=ArrayData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        mode = self.inputs.mode.value
        norder = self.inputs.norder.value
        cwd = self.inputs.cwd.value
        alm_prefix_value = self.inputs.prefix.value

        if mode == "phonons":

            # copy dfset_filename
            if False:
                if isinstance(self.inputs.fcsxml, List):
                    fcsxml_filename = self._FCS_FILENAME
                    with folder.open(fcsxml_filename,  "w", encoding='utf8') as f:
                        f.write("\n".join(self.inputs.fcsxml.get_list()))
                elif isinstance(self.inputs.fcsxml, SinglefileData):
                    fcsxml_filename = self.inputs.fcsxml.list_object_names()[0]
                    with folder.open(fcsxml_filename,  "w", encoding='utf8') as f:
                        f.write(self.inputs.fcsxml.get_content())
                else:
                    raise ValueError("unknown instance in self.inputs.fcsxml. type=", 
                                    type(self.inputs.fcsxml))
            else:
                fcsxml_filename = folder_prepare(folder, self.inputs.fcsxml, self._FCS_FILENAME)

            # make inputfile
            structure = self.inputs.structure.get_ase()

            phonons_mode = self.inputs.phonons_mode.value
            if phonons_mode == "band":
                kpoint_param = make_alm_kpoint(structure, 1)
            elif phonons_mode == "dos":
                if False:
                    if "kspacing" in self.inputs.kparam.attributes:
                        kspacing = self.inputs.kparam.attributes["kspacing"]
                        kpoint_param = make_alm_kpoint(
                            structure, 2, kspacing=kspacing)
                    else:
                        kpoint_param = None
                qmesh_value = self.inputs.qmesh.get_list()
                kpoint_param = ["2", " ".join(list(map(str, qmesh_value)))]
            else:
                raise ValueError("unknown type={type}")

            other_param = self.inputs.param.get_dict()
            if "general" in other_param:
                other_param["general"].update({"FCSXML": fcsxml_filename})
            else:
                other_param["general"] = {"FCSXML": fcsxml_filename}
            if "kpoint" in other_param and kpoint_param is not None:
                other_param["kpoint"].update(kpoint_param)
            else:
                other_param["kpoint"] = kpoint_param

            alm_param = atoms_to_alm_in(mode, structure, dic=other_param,
                                        norder=norder,
                                        prefix=alm_prefix_value)

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

            retrieve_list = [self.options.input_filename,
                             self.options.output_filename]
            if phonons_mode == "band":
                for ext in ["bands"]:
                    filename = f"{alm_prefix_value}.{ext}"
                    retrieve_list.append(filename)
            elif phonons_mode == "dos":
                for ext in ["dos", "thermo"]:
                    filename = f"{alm_prefix_value}.{ext}"
                    retrieve_list.append(filename)

            calcinfo.retrieve_list = retrieve_list
            return calcinfo

        elif mode == "RTA":
            fcsxml = self.inputs.fcsxml
            if False:
                if isinstance(fcsxml, List):
                    # copy dfset_filename
                    target_filename = f'{alm_prefix_value}_RTA.xml'
                    with folder.open(target_filename, "w", encoding='utf8') as f:
                        f.write("\n".join(fcsxml.get_list()))
                elif isinstance(fcsxml, SinglefileData):
                    target_filename = fcsxml.list_object_names()[0]
                    with folder.open(target_filename, "w", encoding='utf8') as f:
                        f.write(fcsxml.get_content())
                else:
                    raise ValueError("unknown instance in self.inputs.fcsxml. type=", 
                                    type(self.inputs.fcsxml))
            else:
                target_filename = folder_prepare(folder, fcsxml, f'{alm_prefix_value}_RTA.xml')

            # make inputfile
            structure = self.inputs.structure.get_ase()

            if False:
                if "kspacing" in self.inputs.kparam.attributes:
                    kspacing = self.inputs.kparam.attributes["kspacing"]
                    kpoint_param = make_alm_kpoint(
                        structure, 2, kspacing=kspacing)
                else:
                    kpoint_param = make_alm_kpoint(structure, 2)

            qmesh_value = self.inputs.qmesh.get_list()
            kpoint_param = ["2", " ".join(list(map(str, qmesh_value)))]

            other_param = self.inputs.param.get_dict()
            if "general" in other_param:
                other_param["general"].update({"FCSXML": target_filename})
            else:
                other_param["general"] = {"FCSXML": target_filename}
            if "kpoint" in other_param:
                other_param["kpoint"].update(kpoint_param)
            else:
                other_param["kpoint"] = kpoint_param

            kappa_spec_value = self.inputs.kappa_spec.value
            if kappa_spec_value > 0:
                if "analysis" in other_param:
                    other_param["analysis"].update(
                        {"KAPPA_SPEC": kappa_spec_value})
                else:
                    other_param["analysis"] = {"KAPPA_SPEC": kappa_spec_value}

            alm_param = atoms_to_alm_in(mode, structure, dic=other_param,
                                        norder=norder,
                                        prefix=alm_prefix_value)

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

            retrieve_list = [self.options.input_filename,
                             self.options.output_filename]

            if kappa_spec_value == 0:
                for ext in ["kl", "result"]:
                    filename = f"{alm_prefix_value}.{ext}"
                    retrieve_list.append(filename)
            elif kappa_spec_value == 1:
                for ext in ["kl", "result", "kl_spec"]:
                    filename = f"{alm_prefix_value}.{ext}"
                    retrieve_list.append(filename)
            else:
                raise ValueError(f"unknown kappa_spec={kappa_spec_value}")

            calcinfo.retrieve_list = retrieve_list
            return calcinfo


def _parse_anphon(handle):
    data = handle.read().splitlines()
    data_iter = iter(data)
    result = {}
    while True:
        line = next(data_iter)
        if line.startswith(" The following files are created:"):
            while True:
                line = next(data_iter).strip()
                if "Phonon band structure" in line:
                    s = line.split()
                    band_filename = s[0]
                    result["phband_filename"] = band_filename
                if "Phonon DOS" in line:
                    s = line.split()
                    dos_filename = s[0]
                    result["phdos_filename"] = dos_filename
                if "Thermodynamic quantities" in line:
                    s = line.split()
                    thermo_filename = s[0]
                    result["thermo_filename"] = thermo_filename
                if len(line) == 0:  # new line
                    break
            if len(result.keys()) > 0:
                return result
            else:
                raise ValueError("failed to get result.")


def _parse_anphon_RTA(handle):
    data = handle.read().splitlines()
    data_iter = iter(data)
    result = {}
    while True:
        line = next(data_iter)
        if "PREFIX" in line:
            s = line.split()
            prefix = s[-1]
            result["prefix"] = prefix
        elif " MODE =" in line:
            s = line.split()
            mode = s[-1]
            result["mode"] = mode
        elif "Lattice thermal conductivity is stored in the file" in line:
            s = line.split()
            kl_filename = s[-1]
            result["kl_filename"] = kl_filename
        elif "Thermal conductivity spectra is stored in the file" in line:
            s = line.split()
            kl_filename = s[-1]
            result["kl_spec_filename"] = kl_filename
        elif "Total Number of phonon modes to be calculated" in line:
            s = line.split()
            nmodes = int(s[-1])
            result["nmodes"] = nmodes
        elif "KAPPA_SPEC =" in line:
            s = line.split()
            kappa_spec = int(s[2])
            result["kappa_spec"] = kappa_spec
        elif "Job finished" in line:
            break

    if len(result.keys()) > 0:
        result["result_filename"] = f"{prefix}.result"
        return result
    else:
        raise ValueError("failed to get result.")


class anphon_ParseJob(Parser):

    def parse(self, **kwargs):
        mode = self.node.inputs.mode.value
        norder = self.node.inputs.norder.value
        prefix = self.node.inputs.prefix.value

        cwd = self.node.inputs.cwd.value
        if len(cwd) > 0:
            os.makedirs(cwd, exist_ok=True)

        if mode == "RTA":
            try:
                output_folder = self.retrieved
            except:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    result = _parse_anphon_RTA(handle=handle)
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            kappa_spec_value = self.node.inputs.kappa_spec.value
            if kappa_spec_value == 0:
                kappa_spec_str = ""
            elif kappa_spec_value == 1:
                kappa_spec_str = "_spec"

            filename = self.node.get_option('input_filename')
            _content = output_folder.get_object_content(filename)
            filename = f"{prefix}_anphon_{mode}{kappa_spec_str}.in"
            target_path = os.path.join(cwd, filename)
            with open(target_path, "w") as f:
                f.write(_content)

            filename = self.node.get_option('output_filename')
            _content = output_folder.get_object_content(filename)
            filename = f"{prefix}_anphon_{mode}{kappa_spec_str}.out"
            target_path = os.path.join(cwd, filename)
            with open(target_path, "w") as f:
                f.write(_content)

            if kappa_spec_value == 0:
                label_list = ["result_filename", "kl_filename"]
            elif kappa_spec_value == 1:
                label_list = ["result_filename",
                              "kl_filename", "kl_spec_filename"]

            for label in label_list:
                filename = result[label]
                _content = output_folder.get_object_content(filename)
                target_path = os.path.join(cwd, filename)
                with open(target_path, "w") as f:
                    f.write(_content)
                self.out(label.replace("filename", "file"),
                         SinglefileData(target_path))

            self.out('results', Dict(dict=result))

        elif mode == "phonons":
            try:
                output_folder = self.retrieved
            except:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    result = _parse_anphon(handle=handle)
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

            phonons_mode = self.node.inputs.phonons_mode.value

            if len(cwd) > 0:
                filename = self.node.get_option('input_filename')
                _content = output_folder.get_object_content(filename)
                filename = f"{prefix}_anphon_{mode}_{phonons_mode}.in"
                target_path = os.path.join(cwd, filename)
                with open(target_path, "w") as f:
                    f.write(_content)

                filename = self.node.get_option('output_filename')
                _content = output_folder.get_object_content(filename)
                filename = f"{prefix}_anphon_{mode}_{phonons_mode}.out"
                target_path = os.path.join(cwd, filename)
                with open(target_path, "w") as f:
                    f.write(_content)

                for label, filename in result.items():
                    _content = output_folder.get_object_content(filename)
                    target_path = os.path.join(cwd, filename)
                    with open(target_path, "w") as f:
                        f.write(_content)
                    self.out(label.replace("filename", "file"),
                             SinglefileData(target_path))

            self.out('results', Dict(dict=result))
