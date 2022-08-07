

class DisplaceArgs:
    def __init__(self,
                 mag=0.02,
                 prefix="disp",
                 QE=None,
                 VASP=None,
                 xTAPP=None,
                 LAMMPS=None,
                 OpenMX=None,
                 pattern_file=None,
                 random=False,
                 random_normalcoord=False,
                 temp=100,
                 ignore_imag=False,
                 every=50,
                 load_mddata=None,
                 prim=None,
                 evec=None,
                 num_disp=None,
                 classical=False,
                 print_disp_stdout=False,
                 pes=None,
                 imag_evec=False,
                 Qrange=None):
        self.mag = mag
        self.prefix = prefix
        self.QE = QE
        self.VASP = VASP
        self.xTAPP = xTAPP
        self.LAMMPS = LAMMPS
        self.OpenMX = OpenMX
        self.pattern_file = pattern_file
        self.random = random
        self.random_normalcoord = random_normalcoord
        self.temp = temp
        self.ignore_imag = ignore_imag
        self.every = every
        self.load_mddata = load_mddata
        self.prim = prim
        self.evec = evec
        self.num_disp = num_disp
        self.classical = classical
        self.print_disp_stdout = print_disp_stdout
        self.pes = pes
        self.imag_evec = imag_evec
        self.Qrange = Qrange

    def vars(self):
        return vars(self)


class ExtractArgs:
    def __init__(self, QE=None,
                 LAMMPS=None,
                 OpenMX=None,
                 VASP=None,
                 emax=None,
                 emin=None,
                 get="disp-force",
                 offset=None,
                 unitname="Rydberg",
                 xTAPP=None,
                 target_file=['disp1.pw.out', 'disp2.pw.out']):
        self.LAMMPS = LAMMPS
        self.OpenMX = OpenMX
        self.QE = QE
        self.VASP = VASP
        self.emax = emax
        self.emin = emin
        self.get = 'disp-force'
        self.offset = offset
        self.target_file = target_file
        self.unitname = unitname
        self.xTAPP = xTAPP

    def vars(self):
        return vars(self)
