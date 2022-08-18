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
from __future__ import annotations


from aiida.plugins import DataFactory

import pandas as pd
import re

StructureData = DataFactory('structure')
SinglefileData = DataFactory('singlefile')
FolderData = DataFactory('folder')
List = DataFactory('list')
ArrayData = DataFactory('array')


def load_anphon_kl(data: str|SinglefileData):
    if isinstance(data, str):
        data = data.splitlines()
    elif isinstance(data, SinglefileData):
        data = data.get_content().splitlines()

    header = data[0]
    v = re.split("[,()]+", header[1:])
    v2 = []
    for _x in v:
        _x = _x.strip()
        v2.append(_x)
    header = v2 
    varname = header[1]
    del header[1]
    unit  = header[-1]
    del header[-1]
    
    varlist = []
    for _x in header[1:]:
        varlist.append(f"{varname}_{_x} {unit}")
    header = [header[0]]
    header.extend(varlist)
        
    lines = []
    for line in data[1:]:
        line = line.strip()
        s = re.split(" +", line)
        v = list(map(float,s))
        lines.append(v)
    return pd.DataFrame(lines, columns=header)


def load_anphon_kl_spec(data: str| SinglefileData):
    if isinstance(data, str):
        data = data.splitlines()
    elif isinstance(data, SinglefileData):
        data = data.get_content().splitlines()

    header = data[0][1:].strip()
    splitted_header = re.split(", *", header)
    unit = splitted_header[-1].split(")")[-1].strip()
    for _i, _x in enumerate(splitted_header):
        if _x == 'Thermal Conductivity Spectra (xx':
            varname = _x
            break
    splitted_header = splitted_header[:_i]

    varname = varname.split("(")[0].strip()
    for ax in ["xx", "yy",  "zz"]:
        splitted_header.append(f"{varname} {ax} {unit}")

    values = []
    for _x in data[1:]:
        _x = _x.strip()
        if len(_x)==0:
            continue
        _xx = re.split(" +", _x.strip())
        values.append(list(map(float,_xx)))

    return  pd.DataFrame(values, columns=splitted_header)
