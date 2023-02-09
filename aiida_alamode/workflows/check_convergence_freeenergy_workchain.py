import os

from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Str,  Dict,  Float
from aiida.engine import WorkChain, ToContext, if_
from aiida.orm import Code
from .check_convergence_free_energy import run_main_


SinglefileData = DataFactory('singlefile')
StructureData = DataFactory('structure')
ArrayData = DataFactory('array')
FrameData = DataFactory('dataframe.frame')


img_band_workchain = WorkflowFactory('alamode.img.band_calculator')
img_dos_workchain = WorkflowFactory('alamode.img.dos_calculator')
img_thermo_workchain = WorkflowFactory('alamode.img.thermo_calculator')


class checkConvergenceFreeenergyWorkchain(WorkChain):
    _FCS_FILENAME = 'anphonon_fcs.xml'
    _RUN_CHAIN = True
    _PREFIX = "alamode"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("super_structure", valid_type=StructureData, help='supercell structure')
        spec.input("primitive_structure", valid_type=StructureData, help='primitive structure')
        spec.input('dfset', valid_type=ArrayData, help='dfset')
        spec.input("history", valid_type=FrameData, required=False, help='history')
        spec.input("params", valid_type=Dict, help='paramsters')
        spec.input("anphon_code", valid_type=Code, help='anphon_code')
        spec.input("cwd", valid_type=Str, required=False,
                   help='directory where files are saved.')

        spec.outline(
            cls.phase1,
            cls.phase2_band_submit,
            cls.phase2_dosthermo_submit,
            cls.phase2_band_validate,
            cls.phase2_dosthermo_validate,
            cls.phase3,
            if_(cls.must_make_img)(
                cls.phase4,
                cls.phase4_validate
            )
        )

        spec.output("rc_best", valid_type=Float, help='the best rc.')
        spec.output("fc", valid_type=ArrayData, help='force constant.')
        spec.output("ols_history", valid_type=FrameData, help='ols history.')
        spec.output("band_results", valid_type=Dict, help='band results.')
        spec.output("bands", valid_type=SinglefileData, help='bands.')
        spec.output("dos", valid_type=ArrayData, help='dos.')
        spec.output("thermo_results", valid_type=Dict, help='thermo results.')
        spec.output("dosthermo_properties", valid_type=ArrayData, help='dos+thermo properties.')
        spec.output("history", valid_type=FrameData, help='updated history.')
        spec.output("results", valid_type=Dict, help='results.')
        spec.output("img_files", valid_type=Dict, help='results.')

    def valid_cwd(self):
        if "cwd" in self.inputs:
            if self.inputs["cwd"] is not None:
                if len(self.inputs["cwd"].value) > 0:
                    return True
        return False

    def phase1(self):
        super_structure = self.inputs.super_structure.get_ase()
        primtive_structure = self.inputs.primitive_structure.get_ase()
        dfset = self.inputs.dfset.get_array('dfset')
        history_data = self.inputs.history.df

        params = self.inputs.params.get_dict()
        rc_best, df_ols_history, fc2, fc_indices = run_main_(super_structure,
                                                             dfset, history_data, primtive_structure, params,
                                                             phase=[1])

        if self.valid_cwd():
            cwd = self.inputs.cwd.value
            from ..io import Fcsxml
            fcsxml = Fcsxml(super_structure.cell, super_structure.get_scaled_positions(),
                            super_structure.numbers)
            fcsxml.set_force_constants(fc2, fc_indices)
            fcsxml.write(os.path.join(cwd, self._FCS_FILENAME))

            df_ols_history.to_csv(os.path.join(cwd, 'ols_history.csv'), index=False)

        self.out('rc_best', Float(rc_best).store())

        array_node = ArrayData()
        array_node.set_array('fc2', fc2)
        array_node.set_array('indices', fc_indices)
        self.ctx.rc_best = rc_best
        self.ctx.fc_node = array_node

        array = ArrayData()
        array.set_array('fc2', fc2)
        array.set_array('indices', fc_indices)
        array.store()
        df_node = FrameData(df=df_ols_history)
        df_node.store()
        self.out('fc', array)
        self.out('ols_history', df_node)

    def phase2_band_submit(self):
        qspacing = 0.1
        conv_thr = 0.1
        temp = 1000
        cv = 2
        ninc = 2
        verbosity = 1

        superstructure_node = self.inputs.super_structure
        primstructure_node = self.inputs.primitive_structure
        fc_node = self.ctx.fc_node
        anphon_code = self.inputs.anphon_code

        if True:
            inputs = {'structure': primstructure_node,
                      'super_structure': superstructure_node,
                      'fc': fc_node,
                      'mode': Str('phonons'),
                      'phonons_mode': Str('band'),
                      'qspacing': Float(qspacing)}
            if 'cwd' in self.inputs:
                inputs.update({'cwd':  self.inputs.cwd})

            band_result = self.submit(anphon_code.get_builder(), **inputs)
        else:
            builder = anphon_code.get_builder()

            builder.structure = primstructure_node
            builder.super_structure = superstructure_node
            builder.fc: fc_node
            builder.mode: Str('phonons')
            builder.phonons_mode: Str('band')
            builder.qspacing: Float(qspacing)
            if 'cwd' in self.inputs:
                builder.cwd = self.inputs.cwd

            band_result = self.submit(builder)

        return ToContext(band=band_result)

    def phase2_band_validate(self):
        assert self.ctx.band.is_finished_ok
        results = self.ctx.band.outputs.results
        self.out('band_results', results)
        bands = self.ctx.band.outputs.bands
        self.out('bands', bands)

    def phase2_dosthermo_submit(self):
        qspacing = 0.1
        conv_thr = 0.1
        temp = 1000
        cv = 2
        ninc = 2
        verbosity = 1

        superstructure_node = self.inputs.super_structure
        primstructure_node = self.inputs.primitive_structure
        fc_node = self.ctx.fc_node
        anphon_code = self.inputs.anphon_code

        inputs = {'structure': primstructure_node,
                  'super_structure': superstructure_node,
                  'fc': fc_node,
                  'mode': Str('phonons'),
                  'phonons_mode': Str('dos+thermo'),
                  'qspacing': Float(qspacing)}
        if 'cwd' in self.inputs:
            inputs.update({'cwd':  self.inputs.cwd})

        thermo_result = self.submit(anphon_code.get_builder(), **inputs)

        return ToContext(thermo=thermo_result)

    def phase2_dosthermo_validate(self):
        assert self.ctx.thermo.is_finished_ok
        results = self.ctx.thermo.outputs.results
        self.out('thermo_results', results)
        properties = self.ctx.thermo.outputs.properties
        self.out('dosthermo_properties', properties)

    def phase3(self):
        super_structure = self.inputs.super_structure.get_ase()
        primtive_structure = self.inputs.primitive_structure.get_ase()
        dfset = self.inputs.dfset.get_array('dfset')
        fc_node = self.ctx.fc_node
        if 'history' in self.inputs:
            history_data = self.inputs.history.df
        else:
            history_data = None
        params = self.inputs.params.get_dict()

        thermo_properties = self.ctx.thermo.outputs.properties
        temperatures = thermo_properties.get_array('temperatures')
        free_energy = thermo_properties.get_array('free_energy')
        rc_best = self.ctx.rc_best
        thermo_result = self.ctx.thermo.outputs.results.get_dict()
        omega_lowest = thermo_result["omega_lowest"]
        ratio_negative = thermo_result["imaginary_ratio"]

        params = self.inputs.params.get_dict()
        status, df_history = run_main_(super_structure, dfset,
                                       history_data, primtive_structure, params,
                                       phase=[3],
                                       rc_best=rc_best, temperatures=temperatures,
                                       free_energy=free_energy, omega_lowest=omega_lowest,
                                       ratio_negative=ratio_negative
                                       )
        self.ctx.phase3_results = status

        self.out('results', Dict(dict=status).store())
        history_node = FrameData(df=df_history).store()
        self.out('history', history_node)

    def must_make_img(self):
        if 'cwd' in self.inputs:
            if self.ctx.phase3_results["converged"]:
                return True
        return False

    def phase4(self):
        primstructure_node = self.inputs.primitive_structure
        bands_node = self.ctx.band.outputs.bands
        dosthermo_properties = self.ctx.thermo.outputs.properties
        if 'cwd' in self.inputs:
            cwd = self.inputs.cwd

        inputs = {"primitive_structure": primstructure_node,
                  'bands': bands_node,
                  'cwd': cwd}
        band_result = self.submit(img_band_workchain, **inputs)
        action = "bands"
        key = f"img_{action}_workchains"
        self.to_context(**{key: band_result})

        inputs = {"primitive_structure": primstructure_node,
                  'dos': dosthermo_properties,
                  'cwd': cwd}
        dos_result = self.submit(img_dos_workchain, **inputs)
        action = "dos"
        key = f"img_{action}_workchains"
        self.to_context(**{key: dos_result})

        inputs = {"primitive_structure": primstructure_node,
                  'thermo_properties': dosthermo_properties,
                  'cwd': cwd}
        thermo_result = self.submit(img_thermo_workchain, **inputs)
        action = "thermo"
        key = f"img_{action}_workchains"
        self.to_context(**{key: thermo_result})

    def phase4_validate(self):
        imgfiles = {}
        for action in ["bands", "dos", "thermo"]:
            key = f"img_{action}_workchains"
            assert self.ctx[key].is_finished_ok
            imgfiles[key] = self.ctx[key].outputs.img.value
        self.out('img_files', Dict(dict=imgfiles).store())
