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
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from ase.io.espresso import read_espresso_in
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
UpfData = DataFactory("upf")
Dict = DataFactory("dict")

def _get_pseudos_from_filename(file_path: str, family_name: str):
    """
    Get pseudos from qe input filename.

    The original code is aiida.orm.nodes.data.upf.get_pseudos_from_structure

    Args:
        file_path (str): qe input file path.
        family_name (str): qe upf family name.
    """
    from aiida.common.exceptions import NotExistent, MultipleObjectsError

    def _get_upfnames_from_filename(file_path: str):
        from alamode.interface.QE import QEParser
        qeparser = QEParser()
        qeparser.load_initial_structure(file_path)
        return qeparser.get_upfs()

    kinds = _get_upfnames_from_filename(file_path)

    pseudo_list = {}
    family_pseudos = {}
    family = UpfData.get_upf_group(family_name)

    for node in family.nodes:
        if isinstance(node, UpfData):
            if node.element in family_pseudos:
                raise MultipleObjectsError(
                    f'More than one UPF for element {node.element} found in family {family_name}'
                )
            family_pseudos[node.element] = node

    for kind, upfname in kinds.items():
        try:
            pseudo_list[kind] = family_pseudos[kind]
        except KeyError:
            raise NotExistent(
                f'No UPF for element {kind} found in family {family_name}')
        pseudo_name = pseudo_list[kind].list_object_names()[0]
        if upfname != pseudo_name:
            raise ValueError(
                f"UPF name {upfname} in the input file doesn't match {pseudo_name} in family {family_name}")

    return pseudo_list


class pwx_CalcJob(CalcJob):
    _WITHMPI = True
    _PSEUDOS_DEFAULT = 'SSSP_1.1_efficiency'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("pwx_input_filename", valid_type=Str)
        #spec.input("pwx_output_filename", valid_type=Str)
        #spec.input("xml_filename", valid_type=Str)
        spec.input("cwd", valid_type=Str)
        spec.input('pseudos', valid_type=Str,
                   default=lambda: Str(cls._PSEUDOS_DEFAULT))
        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.pwx'
        spec.inputs['metadata']['options']['input_filename'].default = 'file.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'file.out'
        spec.input('metadata.options.withmpi',
                   valid_type=bool, default=cls._WITHMPI)
        spec.inputs['metadata']['options']['resources'].default = {'num_machines': 1,
                                                                   'num_mpiprocs_per_machine': 1}
        spec.output('result', valid_type=Dict)
        spec.output('output_file', valid_type=SinglefileData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:

        # pwx input file
        input_file_path = os.path.join(self.inputs.cwd.value,
                                       self.inputs.pwx_input_filename.value)
        print("_file_path", input_file_path, "dest_name",
              self.options.input_filename)
        folder.insert_path(
            input_file_path, dest_name=self.options.input_filename)

        # pseudo potentials
        pseudos = self.inputs.pseudos.value
        for _, upfnode in _get_pseudos_from_filename(input_file_path,
                                                           pseudos).items():
            _upf_filename = upfnode.list_object_names()[0]
            _target_path = os.path.join(self.inputs.cwd.value, _upf_filename)
            with open(_target_path, "w") as f:
                f.write(upfnode.get_content())
                folder.insert_path(_target_path, dest_name=_upf_filename)

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

import math

def _parse_pwx_out_v7_0(data, result):

    total_energy_list = []
    data_iter = iter(data)
    while True:
        try:
            line = next(data_iter)
        except StopIteration:
            is_finished_ok = False
            break
        if "Program PWSCF" in line:
            result["date_start"] =  line.split()[5]
            result["time_start"] = line.split()[-1]
        elif "This run was terminated on:" in line:
            result["date_terminated"] = line.split()[-1]
            result["time_terminated"] = line.split()[-2]
        elif "convergence has been achieved" in line:
            convergence_achieved = True
            result["convergence_achieved"] = convergence_achieved
            result["n_scf_steps"] = int(line.split()[-2])
        elif "convergence NOT achieved after" in line:
            convergence_achieved = False
            result["n_scf_steps"] = int(line.split()[-3])
            
        elif "number of atoms/cell      =" in line:
            result["nat"] = int(line.split()[-1])
        elif "scf convergence threshold =" in line:
            result["etot_conv_thr"] = float(line.split()[-1])
        elif line.startswith("     total energy              ="):
            s = line.split()
            result["total_energy_list"].append(math.fabs(float(s[-2])))
        elif line.startswith("!    total energy              ="):
            s = line.split()
            result["total_energy"] =float(s[-2])
        elif "Forces acting on atoms" in line:
            line = next(data_iter)
            nat = result["nat"]
            forces = []
            for i in range(nat):
                line = next(data_iter)
                s = line.split()
                iatom = int(s[1])
                if iatom != 1+i:
                    raise ValueError(f"inconsistent atom index at {i}, iatom={iatom}")
                force = list(map(float,[s[-3],s[-2],s[-1]]))
                forces.append(force)
            result["forces"] = forces
        elif "JOB DONE." in line:
            result["is_finished_ok"] = True
            break
            
    if len(total_energy_list)>1:
         result["scf_error"]= total_energy_list[-1]- total_energy_list[-2]
    return result

import io

def _parse_pwx_out(filename: str=None, handle: io.TextIOWrapper=None):
    """parse pw.x output file.
    
    is_finished_ok = False if "JOB DONE." isn't found.
    
    Args:
        filename (str, optional): an output filename of pw.x. Defaults to None.
        handle (io.TextIOWrapper, optional): file hander. Defaults to None.
        
    Returns:
        dict: output parameters.
    """
    if handle is None and filename is not None:
        with open(filename) as f:
            data = f.read().splitlines()
    elif handle is not None and filename is None:
        data = handle.read().splitlines()
    
    version = None
    data_iter = iter(data)
    while True:
        try:
            line = next(data_iter)
        except StopIteration:
            is_finished_ok = False
            break
        if "Program PWSCF" in line:
            version = line.split()[2].replace("v.","")
            break
    
    if version is None:
        raise ValueError("No pw.x version is found.")
        
    result = {
    "convergence_achieved" : False,
    "n_scf_steps" : None,
    "forces" : [],
    "nat" : None,
    "etot_conv_thr": None,
    "total_energy_list" : [],
    "total_energy" : None,
    "is_finished_ok" : False,     
    "version": version}
    
    if version== '7.0':
        result = _parse_pwx_out_v7_0(data, result)
    else:
        raise ValueError(f"unknown pw.x version = {version}")
    return result

import xml.etree.ElementTree as ET

def xml_get_array(root, label, astype=float):
    forces = root.find(label)
    # print(forces.attrib)
    lines = []
    for line in forces.text.splitlines():
        line = line.strip()
        if len(line)==0:
            continue
        lines.append(list(map(float,line.split())))
    array = np.array(lines).astype(astype)
    if len(array.shape) != int(forces.attrib["rank"]):
        raise ValueError("xml rank != array in output.forces.")

    dims = list(map(int,forces.attrib["dims"].split()))
    dims.reverse()
    if np.all(np.array(dims)!=np.array(array.shape)):
        raise ValueError("dims != array.shape in output.forces.")
    output_forces = array
    return output_forces

def _xml_get_text(root, label, astype):
    convergence_achieved  = root.find(label)
    if astype==bool:
        if convergence_achieved.text =="true":
            return True
        else:
            return False
    else:
        return astype(convergence_achieved.text)
    
def _xml_get_attrib(root, label, astype):
    label = label.split(",")
    convergence_achieved  = root.find(label[0])
    return astype(convergence_achieved.attrib[label[1]])

def xml_get_value(root, label, astype):
    if "," in label:
        return _xml_get_attrib(root, label, astype)
    else:
        return _xml_get_text(root, label, astype)
    
    
def _parse_pwx_xml_v7_0(root, result):

    nat = xml_get_value(root,
                        "input/atomic_structure,nat", int)
    result["nat"] = nat
    etot_conv_thr = xml_get_value(root,
                  label="input/control_variables/etot_conv_thr", 
                  astype=float)
    result["etot_conv_thr"] = etot_conv_thr
    convergence_achieved = xml_get_value(root,
                  label="output/convergence_info/scf_conv/convergence_achieved", 
                  astype=bool)
    result["convergence_achieved"] = convergence_achieved
    n_scf_steps = xml_get_value(root,
                  label="output/convergence_info/scf_conv/n_scf_steps", 
                  astype=int)
    result["n_scf_steps"] = n_scf_steps
    scf_error = xml_get_value(root,
                  label="output/convergence_info/scf_conv/scf_error", 
                  astype=float)
    result["scf_error"] = scf_error
    
    exit_status = xml_get_value(root,
                  label="exit_status", 
                  astype=int)
    result["exit_status"] = exit_status
    
    if convergence_achieved:
        forces = xml_get_array(root, label="output/forces")
        #result["forces"] = forces
        forces = np.array(forces)*2.0  # unit conversion
        result["forces"] = forces.tolist()
    return result

def _parser_pwx_xml(filename: str):
    """parse pw.x xml file.
    
    Args:
        filename (str): pw.x output filename.
        
    Returns:
        dict:  output parameters.
    """
    with open(filename) as f:
        data = f.read()
    root = ET.fromstring(data)
    
    version = xml_get_value(root,
                        "general_info/creator,VERSION", str)
    print("version", version)

    result = {"version": version}
    if version == "7.0":
        result = _parse_pwx_xml_v7_0(root, result)
    else:
        raise ValueError(f"unknown pw.x version = {version}")
    return result

class pwx_ParseJob(Parser):

    def parse(self, **kwargs):
        try:
            output_folder = self.retrieved
        except:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        try:
            with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                result = _parse_pwx_out(handle=handle)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        except ValueError:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        print("result", result)
        self.out('result', Dict(dict=result))

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
