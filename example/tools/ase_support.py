
from  ase import io
from aiida.orm import Str, Dict, Float, Int
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import calcfunction, workfunction, submit, run


# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')

@calcfunction
def load_atoms(filename: Str, format: Str, style=None) -> StructureData:
    """load atom from file, filename
    
    style must be supplied if format=="lammps-data"
    
    Args:
        filename (Str): filename to read.
        format (Str): filename format.
        
    Returns:
        StructureData: cystal data.
    """
    format = format.value
    filename = filename.value
    if format == "cif":
        CifData = DataFactory('cif')
        cif = CifData(file=filename)
        atoms = cif.ase
    elif format == "lammps-data":
        #atoms = read_lammps_data(filename, style=style)
        atoms = io.read(filename, format=format, style=style)
    elif format== "general":
        atoms = io.read(filename, format=format)
    else:
        print("unknown format", format)
        atoms = io.read(filename, format=format)
    return StructureData(ase=atoms)

