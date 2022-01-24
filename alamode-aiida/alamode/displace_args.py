
class DisplaceArgs:
    def __init__(self, mag=0.01, pattern_file=['si221.pattern_HARMONIC'],
             QE="qe.in",):
        self.LAMMPS=None
        self.OpenMX=None
        self.QE=QE
        self.Qrange=None
        self.VASP=None
        self.classical=False
        self.evec=None
        self.every='50'
        self.ignore_imag=False
        self.imag_evec=False
        self.load_mddata=None
        self.mag=mag
        self.num_disp=1
        self.pattern_file=pattern_file
        self.pes=None
        self.prefix='disp'
        self.prim=None
        self.print_disp_stdout=False
        self.random=False
        self.random_normalcoord=False
        self.temp=100
        self.xTAPP=None
