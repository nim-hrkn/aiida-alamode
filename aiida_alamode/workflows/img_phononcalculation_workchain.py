
from aiida.orm import Str
from aiida.engine import WorkChain
from aiida.plugins import DataFactory

from .compute_phonon_props import PhononCalculator

SinglefileData = DataFactory('singlefile')
StructureData = DataFactory('structure')
ArrayData = DataFactory('array')


class BandImgPhononCalculatorWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("primitive_structure", valid_type=StructureData, help='primitive structure')
        spec.input("bands", valid_type=SinglefileData, help='band file.')
        spec.input("cwd", valid_type=Str, required=False,
                   help='directory where files are saved.')

        spec.outline(
            cls.band_img,
        )

        spec.output("img", valid_type=Str, help='band image')

    def band_img(self):
        primitive_atoms = self.inputs.primitive_structure.get_ase()
        phononcalculator = PhononCalculator(fname_primitive=primitive_atoms)
        cwd = self.inputs.cwd.value
        phononcalculator._cwd = cwd
        bandfile = self.inputs.bands
        img_path = phononcalculator.plot_band_([bandfile])
        self.out('img', Str(img_path).store())


class DosImgPhononCalculatorWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("primitive_structure", valid_type=StructureData, help='primitive structure')
        spec.input("dos", valid_type=ArrayData, help='dos data.')
        spec.input("cwd", valid_type=Str, required=False,
                   help='directory where files are saved.')

        spec.outline(
            cls.band_img,
        )

        spec.output("img", valid_type=Str, help='band image')

    def band_img(self):
        primitive_atoms = self.inputs.primitive_structure.get_ase()
        phononcalculator = PhononCalculator(fname_primitive=primitive_atoms)
        cwd = self.inputs.cwd.value
        phononcalculator._cwd = cwd
        dos = self.inputs.dos.get_array('dos')
        img_path = phononcalculator.plot_dos_(dos)
        self.out('img', Str(img_path).store())


class ThermoImgPhononCalculatorWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("primitive_structure", valid_type=StructureData, help='primitive structure')
        spec.input("thermo_properties", valid_type=ArrayData, help='thermo data.')
        spec.input("cwd", valid_type=Str, required=False,
                   help='directory where files are saved.')

        spec.outline(
            cls.band_img,
        )

        spec.output("img", valid_type=Str, help='band image')

    def band_img(self):
        primitive_atoms = self.inputs.primitive_structure.get_ase()
        phononcalculator = PhononCalculator(fname_primitive=primitive_atoms)
        cwd = self.inputs.cwd.value
        phononcalculator._cwd = cwd
        properties = self.inputs.thermo_properties
        phononcalculator.temperatures = properties.get_array('temperatures')
        phononcalculator.heat_capacity = properties.get_array('heat_capacity')
        phononcalculator.entropy = properties.get_array('entropy')
        phononcalculator.internal_energy = properties.get_array('internal_energy')
        phononcalculator.free_energy = properties.get_array('free_energy')
        phononcalculator.free_energy = properties.get_array('free_energy')
        img_path = phononcalculator._plot_thermo()
        self.out('img', Str(img_path).store())
