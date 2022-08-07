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
from fnmatch import fnmatch
from alamode import plotdos
from alamode import plotband
from aiida.orm import Str, Float, Dict, Int, Bool
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.parsers.parser import Parser
from aiida.engine import CalcJob, calcfunction, WorkChain
from aiida.plugins import DataFactory
#from alamode.extract import check_options, run_parse
from alamode import extract
from alamode.args_defs import ExtractArgs

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


ArrayData = DataFactory('array')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')

if False:
    class displace_Calcjob(CalcJob):
        """displace_Calcjob is obsolte. Please use diplace_*_Calcjob"""
        _DISP_INPUT_FILENAME = 'disp*.pw.in'
        _NORDER = 1

        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("format", valid_type=Str)
            spec.input("structure_org", valid_type=SinglefileData)
            spec.input("mag", valid_type=Float)
            spec.input("pattern_files", valid_type=(Dict, FolderData))
            #spec.input('start_id', valid_type=Int)
            spec.input("cwd", valid_type=Str)
            spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
            spec.input("disp_input_filename", valid_type=Str,
                    default=lambda: Str(cls._DISP_INPUT_FILENAME))
            spec.inputs['metadata']['options']['parser_name'].default = 'alamode.displace'
            spec.inputs['metadata']['options']['input_filename'].default = 'displace.in'
            spec.inputs['metadata']['options']['output_filename'].default = 'displace.out'
            spec.inputs['metadata']['options']['resources'].default = {
                'num_machines': 1, 'num_mpiprocs_per_machine': 1}

            spec.output('result', valid_type=Int)
            spec.output('dispfile_folder', valid_type=FolderData)

        def prepare_for_submission(self, folder: Folder) -> CalcInfo:

            cwd = self.inputs.cwd.value
            structure_org_filename = self.inputs.structure_org.attributes["filename"]
            target_path = os.path.join(cwd, structure_org_filename)
            if not os.path.isfile(target_path):
                with open(target_path, "w") as f:
                    f.write(self.inputs.structure_org.get_content())
            folder.insert_path(target_path, dest_name=structure_org_filename)

            cwd = self.inputs.cwd.value
            if isinstance(self.inputs.pattern_files,Dict):
                for pattern_filename in self.inputs.pattern_files.attributes['pattern_files']:
                    folder.insert_path(os.path.join(cwd, pattern_filename),
                                    dest_name=pattern_filename)
            elif isinstance(self.inputs.pattern_files, FolderData):
                for pattern_filename in self.inputs.pattern_files.list_object_names():
                    folder.insert_path(os.path.join(cwd, pattern_filename),
                                    dest_name=pattern_filename)

            codeinfo = CodeInfo()
            codeinfo.code_uuid = self.inputs.code.uuid
            codeinfo.cmdline_params = [f"--{self.inputs.format.value}={self.inputs.structure_org.attributes['filename']}",
                                    f"--mag={str(self.inputs.mag.value)}", "-pf"]

            for filename in self.inputs.pattern_files.attributes['pattern_files']:
                codeinfo.cmdline_params.append(filename)

            codeinfo.stdout_name = self.options.output_filename

            calcinfo = CalcInfo()
            calcinfo.codes_info = [codeinfo]
            calcinfo.retrieve_list = [self.options.input_filename, self.options.output_filename,
                                    self.inputs.disp_input_filename.value]

            return calcinfo


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


class displace_pf_Calcjob(CalcJob):
    """ displace -pf 
    
    structure_org is assumed to be in the directory, cwd.
    pattern_files is assumed to be in the directory, cwd.

    TODO:
        add List format in pattern_files

    """
    _NORDER = 1
    _PREFIX = "disp"
    _MODE = "pf"

    @classmethod
    def define(cls, spec):
        """initialization
        
        structure_org must be an absolute path.
        """
        super().define(spec)
        spec.input("format", valid_type=Str)
        spec.input("structure_org", valid_type=(SinglefileData, Str))
        spec.input("mag", valid_type=Float)
        spec.input("pattern_files", valid_type=(FolderData, Dict, List))
        spec.input("cwd", valid_type=Str)
        spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX))
        spec.input("mode", valid_type=Str, default=lambda: Str(cls._MODE))

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.displace'
        spec.inputs['metadata']['options']['input_filename'].default = 'displace_pf.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'displace_pf.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('result', valid_type=Dict)
        spec.output('dispfile_folder', valid_type=FolderData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        cwd = self.inputs.cwd.value

        if isinstance(self.inputs.structure_org, Str):
            target_path = self.inputs.structure_org.value
            _, structure_org_filename = os.path.split(target_path)
            folder.insert_path(target_path, dest_name=structure_org_filename)
        elif isinstance(self.inputs.structure_org, SinglefileData):
            structure_org_filename = self.inputs.structure_org.filename
            with folder.open(structure_org_filename, 
                            'w', encoding='utf8') as handle:
                handle.write(self.inputs.structure_org.get_content())

        else:
            raise ValueError("unknown format to self.inputs.structure_org")

        cwd = self.inputs.cwd.value

        if isinstance(self.inputs.pattern_files, Dict):
            for pattern_filename in self.inputs.pattern_files.attributes['pattern_files']:
                folder.insert_path(os.path.join(cwd, pattern_filename),
                                dest_name=pattern_filename)
        elif isinstance(self.inputs.pattern_files, List):
            for pattern_filename in self.inputs.pattern_files.get_list():
                folder.insert_path(os.path.join(cwd, pattern_filename),
                                dest_name=pattern_filename)
        elif isinstance(self.inputs.pattern_files, FolderData):
            for pattern_filename in self.inputs.pattern_files.list_object_names():
                with folder.open(pattern_filename, 
                            'w', encoding='utf8') as handle:
                    handle.write(self.inputs.pattern_files.get_object_content(pattern_filename))

        else:
            raise ValueError("unknown format to self.inputs.pattern_files")

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [f"--{self.inputs.format.value}={structure_org_filename}",
                                   f"--prefix={self.inputs.prefix.value}",
                                   f"--mag={str(self.inputs.mag.value)}", "-pf"]

        if isinstance(self.inputs.pattern_files, Dict):
            for filename in self.inputs.pattern_files.attributes['pattern_files']:
                codeinfo.cmdline_params.append(filename)
        elif isinstance(self.inputs.pattern_files, List):
            for filename in self.inputs.pattern_files.get_list():
                codeinfo.cmdline_params.append(filename)
        elif isinstance(self.inputs.pattern_files, FolderData):
            for filename in self.inputs.pattern_files.list_object_names():
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


class displace_random_Calcjob(CalcJob):

    _NORDER = 1
    _PREFIX = "disp"
    _MODE = "random"

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("format", valid_type=Str)
        spec.input("structure_org", valid_type=(SinglefileData, Str))
        spec.input("mag", valid_type=Float)
        spec.input("num_disp", valid_type=Int)
        spec.input("cwd", valid_type=Str)
        spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input("prefix", valid_type=Str, default=lambda: Str(cls._PREFIX))
        spec.input("mode", valid_type=Str, default=lambda: Str(cls._MODE))

        spec.inputs['metadata']['options']['parser_name'].default = 'alamode.displace'
        spec.inputs['metadata']['options']['input_filename'].default = 'displace_random.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'displace_random.out'
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

        spec.output('result', valid_type=Dict)
        spec.output('dispfile_folder', valid_type=FolderData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:

        cwd = self.inputs.cwd.value

        if isinstance(self.inputs.structure_org, Str):
            target_path = self.inputs.structure_org.value
            _, structure_org_filename = os.path.split(target_path)
            folder.insert_path(target_path, dest_name=structure_org_filename)
        elif isinstance(self.inputs.structure_org, SinglefileData):
            structure_org_filename = self.inputs.structure_org.filename
            with folder.open(self.inputs.structure_org.filename, 
                            'w', encoding='utf8') as handle:
                handle.write(self.inputs.structure_org.get_content())

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

        self.out("result", Dict(dict=output_displace))

        _filename = self.node.get_option('output_filename')
        _content = output_folder.get_object_content(_filename)
        mode = self.node.inputs.mode.value
        prefix = self.node.inputs.prefix.value
        _target_filename = f"displace_{mode}_{prefix}.out"
        target_path = os.path.join(cwd, _target_filename)
        with open(target_path, "w") as f:
            f.write(_content)

        disp_input_filename = _get_str_outfiles(self.node.inputs.format.value,
                                                self.node.inputs.prefix.value)

        folderdata = FolderData()

        for _dispfile_in in output_folder.list_object_names():
            if fnmatch(_dispfile_in, disp_input_filename):
                _content = output_folder.get_object_content(_dispfile_in)
                _dispfile_out = _dispfile_in
                _target_path = os.path.join(cwd, _dispfile_out)
                with open(_target_path, "w") as f:
                    f.write(_content)
                folderdata.put_object_from_file(
                    _target_path, path=_dispfile_out)

        self.out('dispfile_folder', folderdata)


@calcfunction
def _extract(format: Str, structure_org: Str, target_file: List,
             cwd: Str, prefix: Str):

    if True:
        prefix = prefix.value
        output_filename = f"DFSET_{prefix}"
    else:
        norder_value = norder.value
        if norder_value == 1:
            output_filename = "DFSET_harmonic"
        elif norder_value == 2:
            output_filename = "DFSET_cubic"
        else:
            raise ValueError(f"unknown norder={norder}.")

    cwd_value = cwd.value
    array = target_file.get_list()
    _target_file = []
    for _filename in array:
        _target_file.append(os.path.join(cwd_value, _filename))

    # read QE.in from the root directory
    _structure_org = structure_org.value
    inputs = {format.value: _structure_org, "target_file": _target_file }
    # args = ExtractArgs(QE=_structure_org, target_file=_target_file)
    args = ExtractArgs(**inputs)

    code, file_original, output_flags, str_unit = extract.check_options(args)

    _output_filepath = os.path.join(cwd_value, output_filename)

    _stdout_org = sys.stdout
    sys.stdout = open(_output_filepath, "w")
    file_results = args.target_file
    extract.run_parse(args, code, file_original,
                      file_results, output_flags, str_unit)
    sys.stdout.flush()  # necessary
    sys.stdout.close()
    sys.stdout = _stdout_org  # resume stdout

    return SinglefileData(_output_filepath)


class ExtractWorkChain(WorkChain):
    _NORDER = 1
    _PREFIX = "disp"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('format', valid_type=Str)
        spec.input("structure_org", valid_type=Str)
        spec.input("target_file", valid_type=List)
        spec.input("cwd", valid_type=Str)
        # spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input('prefix', valid_type=Str)
        #spec.input("output_filename", valid_type=Str)
        spec.outline(cls.extract)
        spec.output("dfset_file", valid_type=SinglefileData)

    def extract(self):

        output_file = _extract(self.inputs.format,
                               self.inputs.structure_org,
                               self.inputs.target_file,
                               self.inputs.cwd,
                               self.inputs.prefix
                               )

        self.out("dfset_file", output_file)


def _make_phband_figure(files,  unitname: str = "kayser",
                        normalize_xaxis=False, print_key=False,
                        tight_layout=True, filename: str = None):

    nax, xticks_ax, xticklabels_ax, xmin_ax, xmax_ax, ymin, ymax, \
        data_merged_ax = plotband.preprocess_data(
            files, unitname, normalize_xaxis)
    img_filename = plotband.run_plot(files, nax, xticks_ax, xticklabels_ax,
                                     xmin_ax, xmax_ax, ymin, ymax, data_merged_ax,
                                     unitname=unitname, print_key=print_key,
                                     tight_layout=tight_layout, filename=filename, show=False)
    return img_filename


@calcfunction
def _make_band_file(band_filenames: (Str, List, SinglefileData), cwd: Str,
                    prefix: Str, img_filename: Str, unitname: Str):
    if isinstance(band_filenames, SinglefileData):
        _files = band_filenames.list_object_names()
    elif isinstance(band_filenames, List):
        _files = band_filenames.get_list()
    else:
        _files = [band_filenames.value]

    cwd = cwd.value
    img_filename = os.path.join(cwd,
                                "_".join([prefix.value, img_filename.value]))
    files = []
    for _file in _files:
        files.append(os.path.join(cwd, _file))
    img_path = _make_phband_figure(files, unitname.value,
                                   filename=img_filename)
    return SinglefileData(img_path)


class PhbandWorkChain(WorkChain):
    """
    Phonon band workchain.

    band_filenames should support valid_type (SinglefileData, FolderData).
    """
    _UNITNAME_DEFAULT = "kayser"
    _NORDER = 1
    _IMG_FILENAME = "phband.pdf"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str)
        #spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input("prefix", valid_type=Str)
        spec.input("band_filenames", valid_type=(Str, List, SinglefileData))
        spec.input('unitname', valid_type=Str,
                   default=lambda: Str(cls._UNITNAME_DEFAULT))
        spec.input("img_filename", valid_type=Str,
                   default=lambda: Str(cls._IMG_FILENAME))
        spec.outline(cls.make_band_file)
        spec.output("img_file", valid_type=SinglefileData)

    def make_band_file(self):

        img_file = _make_band_file(self.inputs.band_filenames, self.inputs.cwd,
                                   # self.inputs.norder,
                                   self.inputs.prefix,
                                   self.inputs.img_filename, self.inputs.unitname)
        self.out("img_file", img_file)


def _make_phdos_figure(files, unitname="kayser", print_pdos=False,
                       print_key=False, filename: str = None):
    return plotdos.run_plot(files, unitname, print_pdos, print_key, filename=filename,
                            show=False)


@calcfunction
def _make_dos_file(dos_filenames: (Str, List, SinglefileData),
                   cwd: Str, prefix: Str,
                   img_filename: Str, unitname: Str):
    if isinstance(dos_filenames, SinglefileData):
        _files = dos_filenames.list_object_names()
    elif isinstance(dos_filenames, List):
        _files = dos_filenames.get_list()
    else:
        _files = [dos_filenames.value]
    cwd = cwd.value
    files = []
    for _file in _files:
        files.append(os.path.join(cwd, _file))
    target_path = os.path.join(cwd, "_".join(
        [prefix.value, img_filename.value]))
    img_filename = _make_phdos_figure(
        files, unitname.value, filename=target_path)
    return SinglefileData(target_path)


class PhdosWorkChain(WorkChain):
    """
    Phonon DOS workchain.

    dos_filenames should support valid_type (SinglefileData, FolderData).
    """
    _UNITNAME_DEFAULT = "kayser"
    _NORDER = 1
    _IMG_FILENAME = "phdos.pdf"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str)
        #spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input("prefix", valid_type=Str)
        spec.input("dos_filenames", valid_type=(Str, List, SinglefileData))
        spec.input('unitname', valid_type=Str,
                   default=lambda: Str(cls._UNITNAME_DEFAULT))
        spec.input("img_filename", valid_type=Str,
                   default=lambda: Str(cls._IMG_FILENAME))
        spec.outline(cls.make_dos_file)
        spec.output("img_file", valid_type=SinglefileData)

    def make_dos_file(self):
        img_file = _make_dos_file(self.inputs.dos_filenames, self.inputs.cwd,
                                  self.inputs.prefix,
                                  self.inputs.img_filename, self.inputs.unitname)
        self.out("img_file", img_file)


def thermo_load_data(filename: str = None, content: str = None, fmt: str = "df"):
    """load a thermo file and convert to the format.

    if fmt=="df":
        return pd.DataFrame.
    if fmt=="np":
        return np.ndarray, column labels as list.

    Args:
        filename (str, optional): thermo filename. Defaults to None.
        content ([str], optional): content of the thoermo file. Defalts to None.
        fmt (str, optional): output format. Defaults to 'df'

    Returns:
        the contents of the thermo file.
    """
    if filename is not None and content is None:
        with open(filename) as f:
            data = f.read().splitlines()
    elif content is not None:
        data = content.splitlines()

    _header = data[0][1:].split(",")
    header = []
    for _x in _header:
        header.append(_x.strip())
    lines = []
    for line in data[1:]:
        _x = line
        line = list(map(float, _x.split()))
        lines.append(line)
    if fmt == "df":
        df = pd.DataFrame(lines, columns=header)
        return df
    elif fmt == "np":
        return np.array(lines), header
    else:
        raise ValueError(f"unknown format={fmt}.")


@calcfunction
def _make_thermo_figure(thermo_file: (Str, SinglefileData), cwd: Str, prefix: Str,
                        img_filename: Str, show: Bool):
    cwd = cwd.value
    if isinstance(thermo_file, SinglefileData):
        _content = thermo_file.get_content()
    elif isinstance(thermo_file, Str):
        with open(os.path.join(cwd, thermo_file.value)) as f:
            _content = f.read()
    thermo_df = thermo_load_data(content=_content)
    fig, ax = plt.subplots()
    thermo_df.plot(
        x=thermo_df.columns[0], y=thermo_df.columns[-1], legend=None, ax=ax)
    ax.set_ylabel(thermo_df.columns[-1])

    target_path = os.path.join(cwd, "_".join(
        [prefix.value, img_filename.value]))
    fig.tight_layout()

    fig.savefig(target_path)
    if show:
        fig.show()
    plt.close(fig)
    return SinglefileData(target_path)


class FreeenergyImgWorkChain(WorkChain):
    _SHOW_DEFAULT = False
    _NORDER = 1
    _IMG_FILENAME = "phfreeenergy.pdf"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str)
        spec.input("prefix", valid_type=Str)
        spec.input("thermo_file", valid_type=(Str, SinglefileData))
        spec.input("img_filename", valid_type=Str,
                   default=lambda: Str(cls._IMG_FILENAME))
        spec.input("show", valid_type=Bool,
                   default=lambda: Bool(cls._SHOW_DEFAULT))
        spec.outline(cls.make_thermo_fig)
        spec.output("img_file", valid_type=SinglefileData)

    def make_thermo_fig(self):
        img_file = _make_thermo_figure(self.inputs.thermo_file, self.inputs.cwd,
                                       self.inputs.prefix,
                                       self.inputs.img_filename, self.inputs.show)
        self.out("img_file", img_file)
