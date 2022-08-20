import numpy as np


def displacemenpattern_to_lines(displacement_pattern: list) -> list:

    basis = []
    for pats in displacement_pattern:
        for pat in pats:
            basis.append(pat[-1])
    basis = np.array(basis)
    if np.all(basis == 'Cartesian'):
        basis = "C"
    else:
        raise ValueError('not all basis is C')
    lines = [f"Basis : {basis}"]
    for i, pats in enumerate(displacement_pattern):
        lines.append(f'{i+1}: {len(pats)}')
        for pat in pats:
            n = pat[0]
            v = list(map(float, pat[1]))
            lines.append(f' {n+1} {v[0]} {v[1]} {v[2]}')

    return lines


def lines_to_displacementpattern(lines):

    basis_key = lines[0].split(":")[0].strip()
    basis = lines[0].split(":")[1].strip()
    if basis_key == "Basis":
        if basis == "C":
            basis = "Cartesian"
        else:
            raise ValueError(f"unknown basis={basis}")
    else:
        raise ValueError(f"unknown basis_key={basis_key}")

    pattern = []
    lines_iter = iter(lines)
    next(lines_iter)
    while True:
        try:
            line = next(lines_iter)
        except StopIteration:
            break
        n = int(line.split(":")[1].strip())
        nth_pat = []
        for i in range(n):
            line = next(lines_iter)
            s = line.split()
            result = [int(s[0])-1]
            result.append(list(map(float, s[1:])))
            result.append(basis)
            nth_pat.append(result)
        pattern.append(nth_pat)
    return pattern
