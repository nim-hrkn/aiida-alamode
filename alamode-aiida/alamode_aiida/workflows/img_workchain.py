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
from alamode import plotdos
from alamode import plotband
from aiida.orm import Str

from aiida.engine import calcfunction, WorkChain
from aiida.plugins import DataFactory


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ArrayData = DataFactory('array')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')
StructureData = DataFactory('structure')
TrajectoryData = DataFactory('array.trajectory')


def make_pattern_files(displacement_patterns: list, cwd: str, filename_template: str):
    filepath_list = []
    for _i, displacement_pattern in enumerate(displacement_patterns):
        filename = filename_template.replace('{counter}', str(_i))
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


def _make_phband_figure(files,  unitname: str = "kayser",
                        normalize_xaxis=False, print_key=False,
                        tight_layout=True, filename: str = None):

    class _Options:
        emax = None
        emin = None
        print_key = False

    options = _Options()

    nax, xticks_ax, xticklabels_ax, xmin_ax, xmax_ax, ymin, ymax, \
        data_merged_ax = plotband.preprocess_data(
            files, unitname, normalize_xaxis)

    plotband.run_plot(files, nax, xticks_ax, xticklabels_ax,
                      xmin_ax, xmax_ax, ymin, ymax, data_merged_ax,
                      unitname, options.print_key, show=False)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename


@ calcfunction
def _make_band_file(band_filenames: (Str, List, SinglefileData), cwd: Str,
                    prefix: Str, img_filename: Str, unitname: Str):

    cwd = cwd.value
    os.makedirs(cwd, exist_ok=True)

    if isinstance(band_filenames, SinglefileData):
        _file = band_filenames.list_object_names()[0]
        files = [_file]
    elif isinstance(band_filenames, List):
        files = band_filenames.get_list()
    else:
        files = [band_filenames.value]

    filename = img_filename.value.replace('{prefix}', prefix.value)
    img_filepath = os.path.join(cwd, filename)

    filepaths = []
    for _file in files:
        filepaths.append(os.path.join(cwd, _file))

    img_filepath = _make_phband_figure(filepaths, unitname.value,
                                       filename=img_filepath)
    return SinglefileData(img_filepath)


class PhbandWorkChain(WorkChain):
    """
    Phonon band workchain.

    band_filenames should support valid_type (SinglefileData, FolderData).
    """
    _UNITNAME_DEFAULT = "kayser"
    _NORDER = 1
    _IMG_FILENAME = "{prefix}_phband.pdf"

    @ classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str,
                   help='directory where results are saved.')
        # spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input("prefix", valid_type=Str, help='string added to filenames.')
        spec.input("band_file", valid_type=(
            Str, List, SinglefileData), help='phonon band filenames')
        spec.input('unitname', valid_type=Str,
                   default=lambda: Str(cls._UNITNAME_DEFAULT), help='unit of energy')
        spec.input("img_filename", valid_type=Str,
                   default=lambda: Str(cls._IMG_FILENAME), help='image filename')
        spec.outline(cls.make_band_file)
        spec.output("img_file", valid_type=SinglefileData, help='image file')

    def make_band_file(self):

        img_file = _make_band_file(self.inputs.band_file, self.inputs.cwd,
                                   self.inputs.prefix,
                                   self.inputs.img_filename, self.inputs.unitname)
        self.out("img_file", img_file)


def _make_phdos_figure(files, unitname="kayser", print_pdos=False,
                       print_key=False, filename: str = None):
    plotdos.run_plot(files, unitname, print_pdos, print_key, show=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename


@ calcfunction
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
    os.makedirs(cwd, exist_ok=True)

    files = []
    for _file in _files:
        files.append(os.path.join(cwd, _file))
    filename = img_filename.value.replace('{prefix}', prefix.value)
    target_path = os.path.join(cwd, filename)

    target_path = _make_phdos_figure(
        files, unitname.value, filename=target_path)
    return SinglefileData(target_path)


class PhdosWorkChain(WorkChain):
    """
    Phonon DOS workchain.

    dos_filenames should support valid_type (SinglefileData, FolderData).
    """
    _UNITNAME_DEFAULT = "kayser"
    _NORDER = 1
    _IMG_FILENAME = "{prefix}_phdos.pdf"

    @ classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str,
                   help='directory where results are saved.')
        # spec.input("norder", valid_type=Int, default=lambda: Int(cls._NORDER))
        spec.input("prefix", valid_type=Str, help='string added to filenames')
        spec.input("dos_file", valid_type=(
            Str, List, SinglefileData), help='phonon dos')
        spec.input('unitname', valid_type=Str,
                   default=lambda: Str(cls._UNITNAME_DEFAULT), help='unit of energy')
        spec.input("img_filename", valid_type=Str,
                   default=lambda: Str(cls._IMG_FILENAME), help='image filename')
        spec.outline(cls.make_dos_file)
        spec.output("img_file", valid_type=SinglefileData, help='image file')

    def make_dos_file(self):
        img_file = _make_dos_file(self.inputs.dos_file, self.inputs.cwd,
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


@ calcfunction
def _make_thermo_figure(thermo_file: (Str, SinglefileData), cwd: Str, prefix: Str,
                        img_filename: Str):
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

    filename = img_filename.value.replace('{prefix}', prefix.value)
    target_path = os.path.join(cwd, filename)
    fig.tight_layout()
    fig.savefig(target_path)
    plt.close(fig)
    return SinglefileData(target_path)


class FreeenergyImgWorkChain(WorkChain):
    """generate free energy image"""

    _IMG_FILENAME = "{prefix}_phfreeenergy.pdf"

    @ classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str,
                   help='directory where files are saved.')
        spec.input("prefix", valid_type=Str, help='string added to filenames')
        spec.input("thermo_file", valid_type=(
            Str, SinglefileData), help='thermo file')
        spec.input("img_filename", valid_type=Str,
                   default=lambda: Str(cls._IMG_FILENAME), help='image file template')
        spec.outline(cls.make_thermo_fig)
        spec.output("img_file", valid_type=SinglefileData, help='image file')

    def make_thermo_fig(self):
        img_file = _make_thermo_figure(self.inputs.thermo_file, self.inputs.cwd,
                                       self.inputs.prefix,
                                       self.inputs.img_filename)
        self.out("img_file", img_file)
