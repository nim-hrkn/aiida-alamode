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
from ..io.lammps_support import write_lammps_data
from ase import io

from aiida.plugins import DataFactory
from aiida.orm import Str

import os

# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')


def folder_prepare_object(folder, target,
                          actions: list = [List, Str, SinglefileData, StructureData],
                          cwd: (Str, str) = None,
                          filename: str = 'super.structure',
                          format: (Str, str) = None) -> str:
    """put target to the input port in the type of target.

    If target is Str, it must be an absolute path.
    If target is List, it is a list of string.

    Args:
        folder (_type_): folder of calcjob.prepare_submission
        target (_type_): target object.
        actions (list):  a list of object types accepted.
        cwd ((Str,str), optional): the directory where files are saved. Defaults to None.
        filename ((Str,str), optional): filename if target. Defaults to 'super.structure'.
        format ((Str,str)), optional): format of structure. Defaults to None.

    Raises:
        ValueError: if len(filename)==0.
        ValueError: if unknown format.
        TypeError: unknown type of self.inputs.target

    Returns:
        str: filename
    """
    if isinstance(format, Str):
        format = format.value
    if isinstance(cwd, Str):
        cwd = cwd.value

    if isinstance(target, List) and List in actions:
        if target is None:
            raise ValueError('target must not be None.')
        if filename is None or len(filename) == 0:
            raise ValueError('filename must be specified.')
        with folder.open(filename, "w", encoding='utf8') as f:
            f.write("\n".join(target.get_list()))
        return filename

    elif isinstance(target, Str) and Str in actions:
        target_path = target.value
        _, filename = os.path.split(target_path)
        folder.insert_path(target_path, dest_name=filename)

    elif isinstance(target, SinglefileData) and SinglefileData in actions:
        filename = target.list_object_names()[0]
        if len(filename) == 0:
            raise ValueError("len(filename)==0")
        with folder.open(filename,
                         'w', encoding='utf8') as handle:
            handle.write(target.get_content())
        return filename

    elif isinstance(target, StructureData) and StructureData in actions:
        # 1. make target_filenmame in the cwd directory to  e able to read it before running.
        # 2. and add it to folder.insert_path.
        atoms = target.get_ase()
        if isinstance(filename, Str):
            filename = filename.value
        print("filename", filename)

        if filename is None:
            raise ValueError("filename is None")
        if len(filename) == 0:
            raise ValueError("len(filename)==0")
        print("cwd", cwd)
        if cwd is not None and len(cwd) == 0:
            # write into the cwd and then folder.
            target_filepath = os.path.join(cwd, filename)
            if format is None:
                raise ValueError(f'unknown format. format={format}')
            if format == "LAMMPS":
                with open(target_filepath, "w") as f:
                    write_lammps_data(
                        f, atoms, atom_style='atomic', force_skew=True)
            elif format == "QE":
                io.write(target_filepath, style="espresso-in")
            elif format == "VASP":
                io.write(target_filepath, style="vasp")
            else:
                raise ValueError(f'unknown format. format={format}')
            folder.insert_path(target_filepath,
                               dest_name=filename)
            return filename
        else:
            # write into the folder directly.
            with folder.open(filename, 'w', encoding='utf8') as handle:
                if format is None:
                    raise ValueError(f'unknown format. format={format}')
                if format == "LAMMPS":
                    write_lammps_data(
                        handle, atoms, atom_style='atomic', force_skew=True)
                elif format == "QE":
                    io.write(handle, style="espresso-in")
                elif format == "VASP":
                    io.write(handle, style="vasp")
                else:
                    raise ValueError(f'unknown format. format={format}')
            return filename

    else:
        raise TypeError(f"unknown type of target = {type(target)}")


def save_output_folder_files(output_folder, cwd: (Str, str), prefix: (Str, str), except_list: list = []):
    """save files in the output_folder to the cwd directory.

    All the files are saved as {prefix}_{filename}.

    Args:
        output_folder (_type_): output_folder in parseJob.
        cwd (Str, str): the directory where files are saved.
        prefix (Str, str): prefix string.
        except_list (list, optional): a file list which aren't saved. Default to [].

    Returns:
        dict: table of filename -> filename in the cwd directory.
        bool: True always.
    """
    if isinstance(cwd, Str):
        cwd = cwd.value
    if isinstance(prefix, Str):
        prefix = prefix.value

    name_convension = {}
    if len(cwd) > 0:

        os.makedirs(cwd, exist_ok=True)
        # save all the files in to the cwd directory.
        for filename in output_folder.list_object_names():
            if filename not in except_list:
                _content = output_folder.get_object_content(filename)
                name_convension[filename] = prefix+"_"+filename
                filename = prefix+"_"+filename
                target_path = os.path.join(cwd, filename)
                with open(target_path, "w") as f:
                    f.write(_content)

    return name_convension, True


def file_type_conversion(cwd: str, filename: str, output_type):
    """type conversin of the file.

    Args:
        cwd (str): the directory where files are saved.
        filename (str): filename
        output_type (aiida.orm.Data): output type

    Raises:
        TypeError: unknown output_type.
    Returns:
        Tuples containing
        aiida.orm.Data: aiida Data specified by output_type.
        str: error message.
    """
    if output_type == SinglefileData:
        target_path = os.path.join(cwd, filename)
        return SinglefileData(target_path), ''
    elif SinglefileData == List:
        # We already have the file in the cwd folder.
        target_path = os.path.join(cwd, filename)
        try:
            with open(target_path) as f:
                _content = f.read()
        except IOError:
            return None, 'NOFILE'
        force_constants = _content.splitlines()
        return List(list=force_constants), ''
    else:
        raise TypeError(f"unknown outputype={type(output_type)}")
