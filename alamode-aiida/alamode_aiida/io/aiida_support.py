
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Str, Dict, Int
from numpy import isin

# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')

def folder_prepare(folder, 
               target : (List, SinglefileData), 
               filename=None, actions=[List, SinglefileData]):
    
    if isinstance(target, List) and List in actions:
        if filename is None:
            raise ValueError('fcsxml_filename must not be None.')
        with folder.open(filename,  "w", encoding='utf8') as f:
            f.write("\n".join(target.get_list()))
    elif isinstance(target, SinglefileData) and SinglefileData in actions:
        filename = target.list_object_names()[0]
        with folder.open(filename,  "w", encoding='utf8') as f:
            f.write(target.get_content())
    else:
        raise ValueError("unknown instance in node. type=", 
                         type(target))
    return filename

import os

def save_output_folder_files(output_folder, cwd: (Str, str), prefix: (Str, str), except_list: list=[]):
    """save files in the output_folder to the cwd directory.

    All the files are saved as {prefix}_{filename}.

    Args:
        output_folder (_type_): output_folder in parseJob.
        cwd (Str, str): the directory where files are saved.
        prefix (Str, str): prefix string.
        except_list (list, optional): a file list which aren't saved. Default to [].

    Returns:
        dict : table of filename -> filename in the cwd directory.
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

    return name_convension
    