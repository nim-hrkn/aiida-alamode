
import re
import io
from aiida.plugins import DataFactory

SinglefileData = DataFactory('singlefile')

def read_filelike(handle):
    if isinstance(handle, io.TextIOBase):
        data = handle.read().splitlines()
    elif isinstance(handle, str):
        with open(handle) as f:
            data = f.read().splitlines()
    elif isinstance(handle, SinglefileData):
        data = handle.get_content().splitlines()
    return data

def parse_analyze_phonons_kappa_boundary(handler):
    data = read_filelike(handler)

    for line in data:
        if line.startswith("# Size of boundary"):
            s = line.split()
            size = " ".join(s[-2:-1])

    for _i, line in enumerate(data):
        if line.startswith(" "):
            data_start = _i
            break
        lastline = line
    header = lastline

    v = re.split("[,()]+", header[1:])
    v2 = []
    for _x in v:
        _x = _x.strip()
        if len(_x) > 0:
            v2.append(_x)
    header = v2

    varname_unit = header[1]
    del header[1]
    varname_unit = varname_unit.split()

    unit = varname_unit[-1]
    varname = varname_unit[0]

    varlist = []
    for _x in header[1:]:
        varlist.append(f"{varname}_{_x} {unit}")
    header = [header[0]]
    header.extend(varlist)

    values = []
    for line in data[data_start:]:
        line = line.strip()
        s = re.split(" +", line)
        v = list(map(float, s))
        values.append(v)

    return size, header, values


def parse_analyze_phonons_tau_at_temperature(handler):
    data = read_filelike(handler)

    for _i, line in enumerate(data):
        if line.startswith("# Phonon lifetime at temperature"):
            s = line.replace(".", "").split()
            temp = " ".join(s[-2:])
        elif line.startswith("# kpoint range"):
            s = line.replace(".", "").split()
            kpoint_range = s[-2:]
        elif line.startswith("# mode   range"):
            s = line.replace(".", "").split()
            mode_range = s[-2:]
        if line.startswith(" "):
            data_start = _i
            break
        lastline = line

    header = lastline[1:].strip()
    splitted_header = re.split(", *", header)
    unit = splitted_header[-1].split(" ")[-1]
    for _i, _x in enumerate(splitted_header):
        if _x == 'Thermal conductivity par mode (xx':
            varname = _x
            break
    splitted_header = splitted_header[:_i]

    varname = varname.split("(")[0].strip()
    for ax in ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]:
        splitted_header.append(f"{varname} {ax} {unit}")

    values = []
    for _x in data[data_start:]:
        _xx = re.split(" +", _x.strip())
        values.append(list(map(float, _xx)))

    result = {'temp': temp, 'kpoint': kpoint_range, 'mode_range': mode_range}
    return result,  splitted_header,  values


def parse_analyze_phonons_cumulative(handler):
    data = read_filelike(handler)

    for _i, line in enumerate(data):
        if line.startswith("# Cumulative thermal conductivity at temperature"):
            s = line.replace(".", "").split()
            temp = " ".join(s[-2:])
        elif line.startswith("# mode range"):
            s = line.replace(".", "").split()
            mode_range = s[-2:]
        if line.startswith(" "):
            data_start = _i
            break
        lastline = line

    header = lastline[1:].strip()
    splitted_header = re.split(", *", header)
    for _i, _x in enumerate(splitted_header):
        if _x == 'kappa [W/mK] (xx':
            varname = _x
            break
    splitted_header = splitted_header[:_i]

    varname_unit = varname.split()
    varname = varname_unit[0].strip()
    unit = varname_unit[1].strip()
    for ax in ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]:
        splitted_header.append(f"{varname} {ax} {unit}")

    values = []
    for _x in data[data_start:]:
        _xx = re.split(" +", _x.strip())
        values.append(list(map(float, _xx)))

    result = {'temp': temp,  'mode_range': mode_range}
    return result,  splitted_header,  values