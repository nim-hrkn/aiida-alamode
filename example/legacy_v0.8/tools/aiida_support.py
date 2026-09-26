from time import sleep
import os
import distutils
import sys


def wait_for_node_finished(node, sec=2):
    """submitした後に実行終了まで待つ。

    Args:
        node (aiida.orm.Node): submitted node.
        sec (int): wait time in sec.
    """
    while True:
        if node.is_terminated:
            break
        print(f"wait another {sec} sec.")
        sleep(sec)
    print(node.is_finished, node.is_finished_ok)


def get_or_create_local_computer(work_directory, name="localhost"):
    """Retrieve or setup a local computer

    Original node: aiida-lammps

    Parameters
    ----------
    work_directory : str
        path to a local directory for running computations in
    name : str
        name of the computer

    Returns
    -------
    aiida.orm.computers.Computer

    """
    from aiida.common import NotExistent
    from aiida.orm import Computer

    try:
        computer = Computer.objects.get(label=name)
    except NotExistent:
        computer = Computer(
            label=name,
            hostname="localhost",
            description=(
                "localhost computer, " "set up by aiida_lammps tests"),
            transport_type="local",
            scheduler_type="direct",
            workdir=os.path.abspath(work_directory),
        )
        computer.store()
        computer.configure()

    return computer


def get_path_to_executable(executable):
    """Get path to local executable.

    Original node: aiida-lammps

    :param executable: Name of executable in the $PATH variable
    :type executable: str

    :return: path to executable
    :rtype: str

    """
    path = None

    # issue with distutils finding scripts within the python path
    # (i.e. those created by pip install)
    script_path = os.path.join(os.path.dirname(sys.executable), executable)
    if os.path.exists(script_path):
        path = script_path
    if path is None:
        path = distutils.spawn.find_executable(executable)
    if path is None:
        raise ValueError("{} executable not found in PATH.".format(executable))

    return os.path.abspath(path)


def get_or_create_code(entry_point, computer, executable, exec_path=None):
    """Retrieve or setup a local computer

    original coce: aiida-lammps

    Parameters
    ----------
    work_directory : str
        path to a local directory for running computations in
    name : str
        name of the computer

    Returns
    -------
    aiida.orm.computers.Computer

    """
    from aiida.common import NotExistent
    from aiida.orm import Code, Computer

    if isinstance(computer, str):
        computer = Computer.objects.get(label=computer)

    try:
        code = Code.objects.get(
            label="{}".format(entry_point)
        )
    except NotExistent:
        if exec_path is None:
            exec_path = get_path_to_executable(executable)
        code = Code(
            input_plugin_name=entry_point, remote_computer_exec=[
                computer, exec_path]
        )
        code.label = "{}".format(entry_point)
        code.store()

    return code
