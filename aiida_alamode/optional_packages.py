# Copyright 2022 Hiori Kino
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which optional packages (MatterSim, SevenNet-Polar, Equivar, AnisoNet, ...) a computer has.

The plugin needs ALAMODE and aiida-core only; the machine-learning packages are optional and live on the
computer where the ``ase_runner`` code runs.  ``check_on_computer`` runs ``alamode-ase-runner --check``
there through the computer's AiiDA transport (core.local or core.ssh_async alike).
"""


def packages_needed(calculator=None, borninfo_calculator=None, dielectric_model=None) -> list:
    """optional packages implied by the driver options (emt / lj come with ASE)"""
    return [n for n in (calculator, borninfo_calculator, dielectric_model) if n and n not in ("emt", "lj")]


def check_on_computer(code_ase, names) -> dict:
    """{name: {"ok": bool, "detail": str}} from `alamode-ase-runner --check <names>` on the computer of code_ase.

    Raises RuntimeError when the command gives no usable answer (transport failure, or an older
    aiida-alamode there without --check)."""
    names = [n for n in dict.fromkeys(names) if n]
    if not names:
        return {}
    exe = code_ase.get_executable() if hasattr(code_ase, "get_executable") else code_ase.filepath_executable
    command = f"{exe} --check {' '.join(names)}"
    try:
        with code_ase.computer.get_transport() as transport:
            retval, stdout, stderr = transport.exec_command_wait(command)
    except Exception as err:
        raise RuntimeError(f"could not run `{command}` on {code_ase.computer.label}: {err}") from err
    status = {}
    for line in str(stdout).splitlines():
        name, _, rest = line.partition(":")
        if name.strip() in names:
            status[name.strip()] = {"ok": rest.strip().startswith("ok"), "detail": rest.strip()}
    if not status:
        raise RuntimeError(f"`{command}` on {code_ase.computer.label} gave no answer (older aiida-alamode there?): "
                           f"{str(stderr).strip()[:200]}")
    for name in names:
        status.setdefault(name, {"ok": False, "detail": "not reported"})
    return status
