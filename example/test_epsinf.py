"""Check the eps_inf CalcJob (alamode.epsinf_ase, AnisoNet) on known materials through AiiDA.

usage: python test_epsinf.py --computer <label> [--gpu] [--device cuda]
Submits one AseDielectricTensorCalculation per structure, waits, and prints the predicted eps_inf
eigenvalues next to literature values (electronic dielectric constant).
"""
import argparse, os, sys, time
import numpy as np
import ase.io
from ase import Atoms
from ase.build import bulk
import aiida
aiida.load_profile()
from aiida.engine import submit
from aiida.orm import load_code, Str, Dict
from aiida.plugins import CalculationFactory, DataFactory

StructureData = DataFactory('core.structure')
HERE = os.path.dirname(os.path.abspath(__file__))


def cubic(A, B, a):
    return Atoms(symbols=[A, B, "O", "O", "O"], cell=np.eye(3) * a, pbc=True,
                 scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, 0, .5], [0, .5, .5], [.5, .5, 0]])


# name: (atoms, literature eps_inf, reference)
CASES = {
    "Si": (bulk("Si", "diamond", a=5.431), "11.7", "exp."),
    "MgO": (bulk("MgO", "rocksalt", a=4.21), "3.0 (total 9.8)", "exp."),
    "NaCl": (bulk("NaCl", "rocksalt", a=5.64), "2.3 (total 5.9)", "exp."),
    "BaHfO3": (cubic("Ba", "Hf", 4.171), "4.6-4.9", "DFT"),
    "BaZrO3": (cubic("Ba", "Zr", 4.192), "4.9", "DFT"),
    "BaTiO3 cubic": (cubic("Ba", "Ti", 4.00), "5.9-6.7", "DFT (LDA)"),
    "SrTiO3 cubic": (cubic("Sr", "Ti", 3.905), "5.2-6.6", "exp. 5.2 / DFT 6.6"),
    "ZrO2 monoclinic": (ase.io.read(os.path.join(HERE, "ZrO2_P2_1c.cif")), "4.7-5.2 (anisotropic)", "DFT"),
    "TiO2 rutile": (None, "6.8 (perp), 8.4 (par)", "exp."),
}
from ase.spacegroup import crystal
CASES["TiO2 rutile"] = (crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3048, 0.3048, 0)], spacegroup=136,
                                cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90]),) + CASES["TiO2 rutile"][1:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--computer", default=os.environ.get("AIIDA_ALAMODE_COMPUTER", "localhost"))
    p.add_argument("--gpu", action="store_true", help="#SBATCH --gres=gpu:1")
    p.add_argument("--device", default="auto", help="device of the model (auto, cpu, cuda)")
    args = p.parse_args()
    code = load_code(f"ase_runner@{args.computer}")
    options = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 2},
               "max_wallclock_seconds": 1800}
    if args.gpu:
        options["custom_scheduler_commands"] = "#SBATCH --gres=gpu:1"
    nodes = {}
    for name, (atoms, lit, ref) in CASES.items():
        builder = CalculationFactory("alamode.epsinf_ase").get_builder()
        builder.code = code
        builder.structure = StructureData(ase=atoms)
        builder.dielectric_model = Dict({"name": "anisonet", "kwargs": {"device": args.device}})
        builder.metadata.options = options
        builder.metadata.label = f"epsinf {name}"
        nodes[name] = submit(builder)
        print(f"submitted {name}: pk {nodes[name].pk}")
    while not all(n.is_terminated for n in nodes.values()):
        time.sleep(10)
    print(f"\n{'material':16s} {'AnisoNet eps_inf eigenvalues':32s} {'literature':24s} {'ref':18s} {'time [s]':>8s}")
    ok = True
    for name, node in nodes.items():
        atoms, lit, ref = CASES[name]
        if node.is_finished_ok:
            eps = node.outputs.dielectric_tensor.get_array("epsilon_inf")
            w = np.linalg.eigvalsh(eps)
            t = node.outputs.results.get_dict().get("time_total", float("nan"))
            print(f"{name:16s} {str(np.round(w, 2).tolist()):32s} {lit:24s} {ref:18s} {t:8.1f}")
        else:
            ok = False
            print(f"{name:16s} FAILED (pk {node.pk}, exit {node.exit_status}): verdi process report {node.pk}")
    print("computer:", args.computer, "| all finished ok" if ok else "| some failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
