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
from ase import Atoms
from fnmatch import fnmatch

from aiida.orm import Str, Float, Dict, Int, Bool
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.parsers.parser import Parser
from aiida.engine import CalcJob, calcfunction, WorkChain
from aiida.plugins import DataFactory
# from alamode.extract import check_options, run_parse

from ..io.lammps_support import write_lammps_data
from ..io.ase_support import load_atoms_bare
from ..io.displacement import displacemenpattern_to_lines
from ..io.aiida_support import save_output_folder_files

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from ase import io


ArrayData = DataFactory('array')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')
StructureData = DataFactory('structure')
TrajectoryData = DataFactory('array.trajectory')


def _get_str_outfiles(format_: str, prefix: str = "disp", counter: str = "*"):
    if format_ == "VASP":
        code = "VASP"
        struct_format = "VASP POSCAR"
        str_outfiles = f"{prefix}{counter}.POSCAR"

    elif format_ == "QE":
        code = "QE"
        struct_format = "Quantum-ESPRESSO pw.in format"
        str_outfiles = f"{prefix}{counter}.pw.in"

    elif format_ == "xTAPP":
        code = "xTAPP"
        struct_format = "xTAPP cg format"
        str_outfiles = f"{prefix}{counter}.cg"

    elif format_ == "LAMMPS":
        code = "LAMMPS"
        struct_format = "LAMMPS structure format"
        str_outfiles = f"{prefix}{counter}.lammps"

    elif format_ == "OpenMX":
        code = "OpenMX"
        struct_format = "OpenMX dat format"
        str_outfiles = f"{prefix}{counter}.dat"
    else:
        raise ValueError(f"unknown format {format_}")

    return str_outfiles


def make_pattern_files(displacement_patterns: list, cwd: str, filename_template: str):
    filepath_list = []
    for _i, displacement_pattern in enumerate(displacement_patterns):
        filename = filename_template.replace('{counter}',str(_i))
        basis = []
        for pats in displacement_pattern:
            for pat in pats:
                basis.append(pat[-1])
        basis = np.array(basis)
        if np.all(basis == 'Cartesian'):
            basis = "C"
        else:
            raise ValueError('not all basis is C')
        lines = [f"basis : {basis}"]
        for i, pats in enumerate(displacement_pattern):
            lines.append(f'{i+1}: {len(pats)}')
            for pat in pats:
                n = pat[0]
                v = list(map(int, pat[1]))
                lines.append(f' {n+1} {v[0]} {v[1]} {v[2]}')
        # write to the file
        filepath = os.path.join(cwd, filename)
        filepath_list.append(filepath)
    return filepath_list


def _place_structure_org_file(atoms: Atoms,
                              structure_org_filename: str,
                              cwd: str,
                              format: str,
                              ) -> str:

    if len(structure_org_filename) == 0:
        raise ValueError("len(structure_org_filename)==0")
    structure_org_filepath = os.path.join(cwd, structure_org_filename)
    if os.path.isfile(structure_org_filepath):
        return structure_org_filepath

    if format == "LAMMPS":
        with open(structure_org_filepath, "w") as f:
            write_lammps_data(f, atoms, atom_style='atomic', force_skew=True)
    elif format == "QE":
        io.write(structure_org_filepath, style="espresso-in")
    elif format == "VASP":
        io.write(structure_org_filepath, style="vasp")
    else:
        raise ValueError(f'unknown format. format={format}')

    return structure_org_filepath


class displace_ALM_pf_CalcJob(CalcJob):
    """ displace -pf

    structure_org is assumed to be in the directory, cwd.
    pattern_files is assumed to be in the directory, cwd.

    TODO:
        add List format in pattern_files

    """
    _NORDER = 1
    _PREFIX = "disp"
    _MODE = "pf"
    _STRUCTURE_ORG_FILENAME = "super.structure"
    _PARAMETERS = {'LAMMPS': {'lammps-style': 'atomic'}}
    _PATTERN_FILENAME = 'disp.pattern_{counter}'

    @classmethod
    def define(cls, spec):
        """initialization

        structure_org must be an absolute path.
        """
        super().define(spec)
        spec.input("format", valid_type=Str, help='structure file format')
        spec.input("structure_org", valid_type=(
            StructureData, SinglefileData, Str), help='equilibrium structure')

        spec.input("mag", valid_type=Float, help='magnitude of displacement')
        spec.input("displacement_patterns", valid_type=List, help='displacement pattern')
        #spec.input("pattern_filenames", valid_type=List)
        spec.input("cwd", valid_type=Str, help='directory where results are saved.')
        spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER), 
        help='1 (harmonic) or 2 (cubic)')
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX), help='string added to filename.')
        spec.input("mode", valid_type=Str, default=lambda: Str(cls._MODE), help='\'pf\' (fixed)')


        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.displace'
        spec.inputs['metadata']['options']['input_filename'].default = 'displace_pf.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'displace_pf.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)
        spec.output('dispfile_folder', valid_type=FolderData, help='folder containing displacements')
        spec.output('displaced_structures', valid_type=TrajectoryData, help='a set of displaced structures')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        cwd = self.inputs.cwd.value
        atoms = self.inputs.structure_org.get_ase()
        structure_org_filename = self._STRUCTURE_ORG_FILENAME
        format = self.inputs.format.value
        # make structure_org_filenmame in the cwd directory
        # and add it to folder.insert_path.

        if len(structure_org_filename) == 0:
            raise ValueError("len(structure_org_filename)==0")
        structure_org_filepath = _place_structure_org_file(atoms,
                                                           structure_org_filename,
                                                           cwd,
                                                           format)

        folder.insert_path(structure_org_filepath,
                           dest_name=structure_org_filename)

        # pattern_files
        filepath_list = make_pattern_files(self.inputs.displacement_patterns.get_list(),
                                           self.inputs.cwd.value,
                                           self._PATTERN_FILENAME)
        for target_path in filepath_list:
            _, filename = os.path.split(target_path)
            folder.insert_path(target_path, dest_name=filename)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [f"--{self.inputs.format.value}={structure_org_filename}",
                                   f"--prefix={self.inputs.prefix.value}",
                                   f"--mag={str(self.inputs.mag.value)}", "-pf"]

        for filename in self.inputs.filepath_list:
            codeinfo.cmdline_params.append(filename)

        # change self.options.output_filename

        codeinfo.stdout_name = self.options.output_filename

        disp_input_filename = _get_str_outfiles(self.inputs.format.value,
                                                self.inputs.prefix.value)

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.input_filename, self.options.output_filename,
                                  disp_input_filename]

        return calcinfo


class displace_pf_CalcJob(CalcJob):
    """ displace.py -pf

    The pattern files are generated as f"{prefix}.{pattern_file_ext}", where pattern_file_ext is 'harmonic' or 'cubic.
    The displacement files are generated as f"{prefix}{counter}.POSCAR" and so on.

    defalt input filename: displace_Pf.in
    default output filename: displace_pf.out

    """
    _CWD = ""
    _NORDER = 1
    _PREFIX = "disp"
    _MODE = "pf"
    _STRUCTURE_ORG_FILENAME = "super.structure"
    _PARAMETERS = {'LAMMPS': {'lammps-style': 'atomic'}}

    @classmethod
    def define(cls, spec):
        """initialization

        structure_org must be an absolute path.
        """
        super().define(spec)
        spec.input("format", valid_type=Str, help='structure file format')
        spec.input("structure_org", valid_type=StructureData, help='equilibrium structure')

        spec.input("mag", valid_type=Float, help='magnitude of displacement')
        spec.input("pattern", valid_type=List, help='displacement pattern')
        spec.input("cwd", valid_type=Str, default=lambda: Str(cls._CWD), help='directory where results are saved.')
        spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER),help='1 (harmonic) or 2 (cubic)')
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX), help='string added to the filename')
        spec.input("mode", valid_type=Str, default=lambda: Str(cls._MODE), help='displace must (must be \'pf\'')


        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.displace'
        spec.inputs['metadata']['options']['input_filename'].default = 'displace_pf.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'displace_pf.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)
        # spec.output('dispfile_folder', valid_type=FolderData)
        spec.output('displaced_structures', valid_type=TrajectoryData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        cwd = self.inputs.cwd.value
        prefix = self.inputs.prefix.value
        # make structure_org_filenmame in the cwd directory
        # and add it to folder.insert_path.
        atoms = self.inputs.structure_org.get_ase()
        structure_org_filename = self._STRUCTURE_ORG_FILENAME
        if len(structure_org_filename) == 0:
            raise ValueError("len(structure_org_filename)==0")
        format = self.inputs.format.value
        if format == "LAMMPS":
            with folder.open(structure_org_filename, 'w', encoding='utf8') as handle:
                write_lammps_data(
                    handle, atoms, atom_style='atomic', force_skew=True)
        elif format == "QE":
            with folder.open(structure_org_filename, 'w', encoding='utf8') as handle:
                io.write(handle, style="espresso-in")
        elif format == "VASP":
            with folder.open(structure_org_filename, 'w', encoding='utf8') as handle:
                io.write(handle, style="vasp")
        else:
            raise ValueError(f'unknown format. format={self.format.value}')

        pattern_file_ext_list = ["harmonic", "cubic"]
        for pattern_file_ext, content in zip(pattern_file_ext_list, self.inputs.pattern.get_list()):
            pattern_filename = f"{prefix}.{pattern_file_ext}"
            lines = displacemenpattern_to_lines(content)
            with folder.open(pattern_filename, 'w', encoding='utf8') as handle:
                handle.write("\n".join(lines))

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [f"--{self.inputs.format.value}={structure_org_filename}",
                                   f"--prefix={self.inputs.prefix.value}",
                                   f"--mag={str(self.inputs.mag.value)}", "-pf"]

        for pattern_file_ext, content in zip(pattern_file_ext_list, self.inputs.pattern.get_list()):
            pattern_filename = f"{prefix}.{pattern_file_ext}"
            codeinfo.cmdline_params.append(pattern_filename)

        # change self.options.output_filename

        codeinfo.stdout_name = self.options.output_filename

        disp_input_filename = _get_str_outfiles(self.inputs.format.value,
                                                self.inputs.prefix.value)

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.input_filename, self.options.output_filename,
                                  disp_input_filename]

        return calcinfo


class displace_random_CalcJob(CalcJob):
    """ displace.py --random

    The displacement files are generated as f"{prefix}{counter}.POSCAR" and so on.

    default input filename: displace_random.in
    default output filename: displace_random.out
    """
    _NORDER = 1
    _PREFIX = "disp"
    _MODE = "random"
    _STRUCTURE_ORG_FILENAME = "super.structure"
    _PARAMETERS = {'LAMMPS': {'lammps-style': 'atomic'}}

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("format", valid_type=Str, help='structure file format')
        spec.input("structure_org", valid_type=(
            StructureData, SinglefileData, Str), help='equilibrium structure')

        spec.input("mag", valid_type=Float, help='magnitude of displacement')
        spec.input("num_disp", valid_type=Int, help='number of set of displacement')
        spec.input("cwd", valid_type=Str, help='directory where results are saved.')
        spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER),help='1 (harmonic) or 2 (cubic)')
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX), help='string added to filenames.')
        spec.input("mode", valid_type=Str, default=lambda: Str(cls._MODE), help='\'random\' (fixed)')


        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.displace'
        spec.inputs['metadata']['options']['input_filename'].default = 'displace_random.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'displace_random.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('results', valid_type=Dict)
        spec.output('dispfile_folder', valid_type=FolderData)
        spec.output('displaced_structures', valid_type=TrajectoryData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:

        cwd = self.inputs.cwd.value
        if len(cwd)>0:
            os.makedirs(cwd, exist_ok=True)

        if isinstance(self.inputs.structure_org, Str):
            target_path = self.inputs.structure_org.value
            _, structure_org_filename = os.path.split(target_path)
            folder.insert_path(target_path, dest_name=structure_org_filename)
        elif isinstance(self.inputs.structure_org, SinglefileData):
            structure_org_filename = self._STRUCTURE_ORG_FILENAME
            with folder.open(self.inputs.structure_org.filename,
                             'w', encoding='utf8') as handle:
                handle.write(self.inputs.structure_org.get_content())
        elif isinstance(self.inputs.structure_org, StructureData):
            # make structure_org_filenmame in the cwd directory
            # and add it to folder.insert_path.
            atoms = self.inputs.structure_org.get_ase()
            structure_org_filename = self._STRUCTURE_ORG_FILENAME
            if len(structure_org_filename) == 0:
                raise ValueError("len(structure_org_filename)==0")
            structure_org_filepath = os.path.join(cwd, structure_org_filename)
            format = self.inputs.format.value
            if format == "LAMMPS":
                with open(structure_org_filepath, "w") as f:
                    write_lammps_data(
                        f, atoms, atom_style='atomic', force_skew=True)
            elif format == "QE":
                io.write(structure_org_filepath, style="espresso-in")
            elif format == "VASP":
                io.write(structure_org_filepath, style="vasp")
            else:
                raise ValueError(f'unknown format. format={self.format.value}')
            folder.insert_path(structure_org_filepath,
                               dest_name=structure_org_filename)
        else:
            raise ValueError("unknown instance to self.inputs.structure_org")

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [f"--{self.inputs.format.value}={structure_org_filename}",
                                   f"--prefix={self.inputs.prefix.value}",
                                   f"--mag={str(self.inputs.mag.value)}",
                                   "--random", f"--num_disp={self.inputs.num_disp.value}"
                                   ]
        codeinfo.stdout_name = self.options.output_filename

        disp_input_filename = _get_str_outfiles(self.inputs.format.value,
                                                self.inputs.prefix.value)

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.input_filename, self.options.output_filename,
                                  disp_input_filename]

        return calcinfo


def _parse_displace(handle):
    data = handle.read().splitlines()
    data_iter = iter(data)
    displacement = {}
    while True:
        line = next(data_iter)
        if "Displacement mode" in line:
            s = line.split(":")
            displacement_mode = {"displacement_mode": s[1].strip()}
            displacement.update(displacement_mode)
        elif "Output file names" in line:
            s = line.split(":")
            output_filename = {"output_filename": s[1].strip()}
            displacement.update(output_filename)
        elif "Number of displacements" in line:
            s = line.split(":")
            number_of_displacement = {
                "number_of_displacements": int(s[1].strip())}
            displacement.update(number_of_displacement)
            return displacement
    return None


def _read_structure(structure_filepath, format, style=None):
    if format == "LAMMPS":
        atoms = io.read(structure_filepath, style='lammps-data',
                        atom_style=style)
    elif format == "QE":
        atoms = io.read(structure_filepath, style="espresso-in")
    elif format == "VASP":
        atoms = io.read(structure_filepath, style="vasp")
    else:
        raise ValueError(f'unknown format. format={format}')
    return atoms


class displace_ParseJob(Parser):

    def parse(self, **kwargs):
        """_target_filename = f"displace_{mode}_{prefix}.out"

        Returns:
            _type_: _description_
        """
        try:
            output_folder = self.retrieved
        except:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        try:
            with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
                output_displace = _parse_displace(handle=handle)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        except ValueError:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        cwd = self.node.inputs.cwd.value
        if len(cwd)>0:
            os.makedirs(cwd, exist_ok=True)

        self.out("results", Dict(dict=output_displace))



        disp_input_filename = _get_str_outfiles(self.node.inputs.format.value,
                                                self.node.inputs.prefix.value)

        folderdata = FolderData()
        format = self.node.inputs.format.value
        if format == "LAMMPS":
            io_format = 'lammps-data'
        elif format == "QE":
            io_format = "espresso-in"
        elif format == "VASP":
            io_format = "vasp"
        else:
            raise ValueError(f'unknown format. format={self.format.value}')

        import io

        displaced_structures = []
        for _dispfile_in in output_folder.list_object_names():
            if fnmatch(_dispfile_in, disp_input_filename):
                _content = output_folder.get_object_content(_dispfile_in)
                # read as ase Atoms
                atoms = load_atoms_bare(
                    io.StringIO(_content), io_format)
                displaced_structures.append(StructureData(ase=atoms))

        if False:
            if len(cwd) > 0:
                for _dispfile_in in output_folder.list_object_names():
                    _content = output_folder.get_object_content(_dispfile_in)
                    _dispfile_out = _dispfile_in
                    _target_path = os.path.join(cwd, _dispfile_out)
                    with open(_target_path, "w") as f:
                        f.write(_content)
        else:
            save_output_folder_files(output_folder, cwd, self.node.inputs.prefix.value)

        self.out('displaced_structures', TrajectoryData(displaced_structures))

