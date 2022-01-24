class ExtractArgs:
    def __init__(self, QE=None, target_file=['disp1.pw.out', 'disp2.pw.out']):
        self.LAMMPS=None
        self.OpenMX=None
        self.QE=QE
        self.VASP=None
        self.emax=None
        self.emin=None
        self.get='disp-force'
        self.offset=None
        self.target_file=target_file
        self.unitname='Rydberg'
        self.xTAPP=None

