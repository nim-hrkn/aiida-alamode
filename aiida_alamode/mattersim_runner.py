"""backward-compatible alias of aiida_alamode.ase_runner (the runner is not MatterSim specific)."""
from .ase_runner import *  # noqa: F401,F403
from .ase_runner import main, CALCULATORS, calculator_spec, make_calculator, run  # noqa: F401
