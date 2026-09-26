#!/usr/bin/env python
# coding: utf-8


from ase import io
from aiida_alamode.workflows.model_selection import ForceConstantModelSelector
import os

base_folder = os.getcwd()

dfset_file = os.path.join(base_folder, "DFSET_random_latest")
poscar_file = os.path.join(base_folder, "SPOSCAR")


format = "vasp"
supercell = io.read(poscar_file, format=format)
atomic_numbers = supercell.get_atomic_numbers()
numbers = len(atomic_numbers)
numbers


with open(poscar_file) as f:
    supercell = io.read(f, format=format)
supercell


with open(dfset_file) as f:
    lines = f.read()
lines = lines.split("\n")


dfset_block = []
content = []
for line in lines:
    if line.startswith("#"):
        if len(content) > 0:
            dfset_block.append(content)
            content = []
    content.append(line)

if len(content) > 0:
    dfset_block.append(content)
    content = []


nadd = 2
step = 0
final_block = 0
while True:
    blocks_write = []
    final_block += nadd
    blocks_write = dfset_block[:final_block]
    print("choose upto", final_block)
    step_string = "{:04d}".format(step+1)
    os.makedirs(os.path.join(base_folder, step_string), exist_ok=True)
    filepath = os.path.join(base_folder, step_string, "dfset")
    print(filepath)
    with open(filepath, "w") as f:
        for block in blocks_write:
            for line in block:
                f.write(line+"\n")
    if final_block > len(dfset_block):
        break
    step += 1


forceconstant = ForceConstantModelSelector(poscar_file)
for step in range(13):
    step_string = "{:04d}".format(step+1)
    filepath = os.path.join(base_folder, step_string, "dfset")
    forceconstant.load_training_data(fname_dfset=filepath)
