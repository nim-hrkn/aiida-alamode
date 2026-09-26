"""Which optional packages (ML force fields, Z* and eps_inf models) are available, locally or on an AiiDA computer.

The plugin itself needs only ALAMODE and aiida-core; the examples use MatterSim (forces), SevenNet-Polar or
Equivar (Z*) and AnisoNet (eps_inf).  Run this before the examples:

    python check_packages.py                                   # this Python
    python check_packages.py --computer host                   # where the code ase_runner@host runs
    python check_packages.py --computer X --require mattersim sevennet-polar anisonet   # exit 1 if one is missing

Prints one line per package: `name: ok` or `name: missing (...)`.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--computer", default=os.environ.get("AIIDA_ALAMODE_COMPUTER"),
                        help="AiiDA computer label: check the Python of the code ase_runner@<computer> (env AIIDA_ALAMODE_COMPUTER); "
                             "without it, this Python")
    parser.add_argument("--require", nargs="*", default=[], metavar="NAME", help="exit 1 unless these are all available")
    parser.add_argument("names", nargs="*", help="packages to check (default: all known)")
    args = parser.parse_args()
    names = list(dict.fromkeys(args.names + args.require))
    if args.computer:
        import aiida
        aiida.load_profile()
        from aiida.orm import load_code
        sys.path.insert(0, HERE)
        from run_alamode_phonons import check_optional_packages
        from aiida_alamode.ase_runner import PACKAGES
        status = check_optional_packages(load_code(f"ase_runner@{args.computer}"), names or list(PACKAGES), exit_on_missing=False)
    else:
        from aiida_alamode.ase_runner import check_packages
        result = check_packages(names or None)
        status = {}
        for name, r in result.items():
            status[name] = r["ok"]
            print(f"{name}: {'ok' if r['ok'] else 'missing (' + '; '.join(r['missing']) + ')'}")
    missing = [n for n in args.require if not status.get(n, False)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
