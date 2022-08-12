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
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory

import os
from fnmatch import fnmatch

from .alm_input import make_alm_in, atoms_to_alm_in, make_alm_kpoint

AU2ANG = 0.529177


StructureData = DataFactory('structure')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')
ArrayData = DataFactory('array')


def _parse_alm_suggest(filename=None, handle=None):
    if filename is not None and handle is None:
        with open(filename) as f:
            data = f.read().splitlines()
    elif filename is None and handle is not None:
        data = handle.read().splitlines()

    num_disp = {}
    num_free_fcs = {}
    disp_pattern = {}
    data_iter = iter(data)
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
                disp_pattern[s[0].strip()] = s[-1].strip()
        elif "Job finished" in line:
            break
    return {"num_free_fcs,": num_free_fcs, "num_disp": num_disp,
            "disp_pattern": disp_pattern}


def _alm_pattern_file(prefix, norder):
    if norder == 1:
        term = "HARMONIC"
    elif norder == 2:
        term = "ANHARM3"
    else:
        raise ValueError(f"uknown order={norder}")
    return f"{prefix}.pattern_{term}"


def _alm_suggest_retrieve_pattern_file(retrieved: Folder, prefix: Str,
                                       cwd: Str, norder: Int) -> FolderData:
    cwd_value = cwd.value
    prefix_value = prefix.value
    folderdata = FolderData()
    for filename in retrieved.list_object_names():
        if fnmatch(filename, f"{prefix_value}.pattern_*"):
            _content = retrieved.get_object_content(filename)
            target_path = os.path.join(cwd_value, filename)

            with open(target_path, "w") as f:
                f.write(_content)
            folderdata.put_object_from_file(target_path, path=filename)
    return folderdata


def _parse_alm_opt(handle):
    data = handle.read().splitlines()

    data_iter = iter(data)
    result = {}
    constraint = {}
    optimization = {}
    outputfiles = {}
    warning_messages = []
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
            break
    results = {"constraint": constraint, "optimization": optimization,
               "outputfiles": outputfiles}
    if len(warning_messages) > 0:
        results["warning"] = warning_messages
    return results


class almBaseCalcJob(CalcJob):
    """
    If fc2xml_file is SinglefileData, first place it in cwd directory with the same name as SinglefileData.
    """
    _WITHMPI = True

    _PREFIX_DEFAULT = "alamode"
    _CUTOFF_DEFAULT = {"*-*": [None]}
    _PARAM_DEFAULT = {}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("prefix", valid_type=Str,
                   default=lambda: Str(cls._PREFIX_DEFAULT))
        spec.input("cwd", valid_type=Str)
        spec.input("norder", valid_type=Int)

        spec.input('cutoff', valid_type=Dict,
                   default=lambda: Dict(dict=cls._CUTOFF_DEFAULT))

        spec.input('param', valid_type=Dict,
                   default=lambda: Dict(dict=cls._PARAM_DEFAULT))

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.alm'
        spec.inputs['metadata']['options']['input_filename'].default = 'alm.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'alm.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)


class almSuggestCalcJob(almBaseCalcJob):
    """alm mode="suggest"
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
        spec.input('mode', valid_type=Str, default=lambda: Str(cls._MODE))
        spec.inputs['metadata']['options']['input_filename'].default = 'alm_suggest.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'alm_suggest.out'        
        spec.expose_outputs(almBaseCalcJob)
        spec.output('pattern_folder', valid_type=FolderData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        cwd = self.inputs.cwd.value
        mode = self.inputs.mode.value
        alm_prefix_value = self.inputs.prefix.value
        norder = self.inputs.norder.value

        pattern_files = []
        for iorder in range(1, norder+1):
            pattern_file = _alm_pattern_file(alm_prefix_value, iorder)
            pattern_files.append(pattern_file)

        # make inputfile
        structure = self.inputs.structure.get_ase()
        cutoff_value = self.inputs.cutoff.get_dict()

        alm_param = atoms_to_alm_in(
            "suggest", structure, prefix=alm_prefix_value, norder=norder,
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
    """
    _WITHMPI = True
    _MODE = "opt"
    _DFSET_FILE = ""
    _FC2XML_FILE = ""
    _PREFIX_DEFAULT = "alamode"
    _CUTOFF_DEFAULT = {"*-*": [None]}
    _PARAM_DEFAULT = {}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(almOptCalcJob)
        spec.input('mode', valid_type=Str, default=lambda: Str(cls._MODE))
        spec.input('dfset_file', valid_type=(Str, SinglefileData),
                   default=lambda: Str(cls._DFSET_FILE))
        spec.input('fc2xml_file', valid_type=(Str, SinglefileData),
                   default=lambda: Str(cls._FC2XML_FILE))
        spec.inputs['metadata']['options']['input_filename'].default = 'alm_opt.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'alm_opt.out' 
        spec.expose_outputs(almOptCalcJob)
        # spec.output('pattern_folder', valid_type=FolderData)
        spec.output('input_ANPHON_file', valid_type=SinglefileData)
        spec.output('force_constants_file', valid_type=SinglefileData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        cwd = self.inputs.cwd.value
        mode = self.inputs.mode.value
        alm_prefix_value = self.inputs.prefix.value
        norder = self.inputs.norder.value

        # copy dfset_filename
        DFSETfilename = self.inputs.dfset_file.attributes["filename"]
        target_path = os.path.join(cwd, DFSETfilename)
        if not os.path.isfile(target_path):
            with open(target_path, "w") as f:
                f.write(self.inputs.dfset_file.get_content())
        folder.insert_path(os.path.join(cwd, DFSETfilename),
                           dest_name=DFSETfilename)

        if isinstance(self.inputs.fc2xml_file, SinglefileData):
            fc2xmlfilename = self.inputs.fc2xml_file.attributes["filename"]
            target_path = os.path.join(cwd, fc2xmlfilename)
            if not os.path.isfile(target_path):
                # You can make it only in the folder directory,
                # but force making in the cwd directory.
                with open(target_path, "w") as f:
                    f.write(self.inputs.fc2xml_file.get_content())
            folder.insert_path(os.path.join(cwd, fc2xmlfilename),
                               dest_name=fc2xmlfilename)
        elif isinstance(self.inputs.fc2xml_file, Str):
            _target_path = self.inputs.fc2xml_file.value
            if len(_target_path) > 0:
                _, _fc2xmlfilename = os.path.split(_target_path)
                folder.insert_path(_target_path, dest_name=_fc2xmlfilename)

        # make inputfile
        structure = self.inputs.structure.get_ase()
        param = self.inputs.param.get_dict()
        if "optimize" in param.keys():
            param["optimize"]["DFSET"] = DFSETfilename
        else:
            param["optimize"] = {"DFSET": DFSETfilename}
        if isinstance(self.inputs.fc2xml_file, SinglefileData):
            param["optimize"]["FC2XML"] = self.inputs.fc2xml_file.list_object_names()[
                0]
        if isinstance(self.inputs.fc2xml_file, Str) and len(self.inputs.fc2xml_file.value) > 0:
            target_path = self.inputs.fc2xml_file.value
            _, fc2xmlfilename = os.path.split(target_path)
            param["optimize"]["FC2XML"] = fc2xmlfilename  # only basename

        alm_param = atoms_to_alm_in("opt", structure,
                                    dic=param, norder=norder,
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
        calcinfo.retrieve_list = [
            self.options.input_filename, self.options.output_filename]

        for ext in ["fcs", "xml"]:
            filename = f"{alm_prefix_value}.{ext}"
            calcinfo.retrieve_list.append(filename)

        return calcinfo


class alm_ParseJob(Parser):

    def parse(self, **kwargs):

        mode = self.node.inputs.mode.value
        cwd = self.node.inputs.cwd.value
        alm_prefix_node = self.node.inputs.prefix
        prefix = self.node.inputs.prefix.value

        if mode == "optimize":
            mode = "opt"

        if mode == "suggest":
            try:
                output_folder = self.retrieved
            except:
                return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

            try:
                with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                    result = _parse_alm_suggest(handle=handle)
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT


            filelist = [self.node.get_option('input_filename'),
                        self.node.get_option('output_filename')]
            local_filelist = [
                f"alm_{mode}_{prefix}.in", 
                f"alm_{mode}_{prefix}.out"]
            for filename, localfilename in zip(filelist, local_filelist):
                if filename in output_folder.list_object_names():
                    _content = output_folder.get_object_content(filename)
                    target_path = os.path.join(cwd, localfilename)
                    with open(target_path, "w") as f:
                        f.write(_content)
                else:
                    raise ValueError(
                        f"no filename={filename} in retrieved data.")

            pattern_folder = _alm_suggest_retrieve_pattern_file(output_folder,
                                                                alm_prefix_node,
                                                                self.node.inputs.cwd,
                                                                self.node.inputs.norder)

            self.out('results', Dict(dict=result))
            self.out('pattern_folder', pattern_folder)

        elif mode == "opt":
            try:
                output_folder = self.retrieved
            except:
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
            local_filelist = [
                f"alm_{mode}_{prefix}.in", 
                f"alm_{mode}_{prefix}.out"]
            for filename, localfilename in zip(filelist, local_filelist):
                if filename in output_folder.list_object_names():
                    _content = output_folder.get_object_content(filename)
                    target_path = os.path.join(cwd, localfilename)
                    with open(target_path, "w") as f:
                        f.write(_content)
                else:
                    raise ValueError(
                        f"no filename={filename} in retrieved data.")

            filename = result["outputfiles"]["input_ANPHON"]
            _content = output_folder.get_object_content(filename)
            target_path = os.path.join(cwd, filename)
            with open(target_path, "w") as f:
                f.write(_content)
            input_anphon_file = SinglefileData(target_path)

            filename = result["outputfiles"]["force_constants"]
            _content = output_folder.get_object_content(filename)
            target_path = os.path.join(cwd, filename)
            with open(target_path, "w") as f:
                f.write(_content)
            force_constants_file = SinglefileData(target_path)

            self.out("input_ANPHON_file", input_anphon_file)
            self.out("force_constants_file", force_constants_file)

            self.out('results', Dict(dict=result))


class anphon_CalcJob(CalcJob):
    _WITHMPI = False
    _NORDER = 1  # dummy
    _PREFIX_DEFAULT = "alamode"
    _MODE = "phonon"
    _PHONONS_MODE = 'dos'
    _PARAM = {}
    _KAPPA_SPEC = 0
    _QMESH_LIST = [20, 20, 20]

    @classmethod
    def define(cls, spec):

        super().define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("prefix", valid_type=Str,
                   default=lambda: Str(cls._PREFIX_DEFAULT))
        spec.input("cwd", valid_type=Str)
        spec.input('norder', valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input('fcsxml', valid_type=SinglefileData)
        spec.input('mode', valid_type=Str, default=lambda: Str(cls._MODE))
        spec.input('phonons_mode', valid_type=Str,
                   default=lambda: Str(cls._PHONONS_MODE))
        spec.input('kappa_spec', valid_type=Int,
                   default=lambda: Int(cls._KAPPA_SPEC))
        # spec.input('kparam', valid_type=Dict, default=lambda: Dict(dict=cls._PARAM))
        spec.input('qmesh', valid_type=List,
                   default=lambda: List(list=cls._QMESH_LIST))
        spec.input('param', valid_type=Dict,
                   default=lambda: Dict(dict=cls._PARAM))

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

        if mode == "phonons":

            # copy dfset_filename
            fcsxml = self.inputs.fcsxml.attributes["filename"]
            target_path = os.path.join(cwd, fcsxml)
            if not os.path.isfile(target_path):
                with open(target_path, "w") as f:
                    f.write(self.inputs.fcsxml.get_content())
            folder.insert_path(target_path,
                               dest_name=fcsxml)

            # make inputfile
            structure = self.inputs.structure.get_ase()
            alm_prefix_value = self.inputs.prefix.value

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
                other_param["general"].update({"FCSXML": fcsxml})
            else:
                other_param["general"] = {"FCSXML": fcsxml}
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

            # copy dfset_filename
            fcsxml = self.inputs.fcsxml.attributes["filename"]
            target_path = os.path.join(cwd, fcsxml)
            if not os.path.isfile(target_path):
                with open(target_path, "w") as f:
                    f.write(self.inputs.fcsxml.get_content())
            folder.insert_path(target_path,
                               dest_name=fcsxml)

            # make inputfile
            structure = self.inputs.structure.get_ase()
            alm_prefix_value = self.inputs.prefix.value

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
                other_param["general"].update({"FCSXML": fcsxml})
            else:
                other_param["general"] = {"FCSXML": fcsxml}
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

            cwd = self.node.inputs.cwd.value

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

            cwd = self.node.inputs.cwd.value
            phonons_mode = self.node.inputs.phonons_mode.value

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
