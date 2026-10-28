## This code patches some missing assertions in mpp

import pathlib

scripts_to_patch = ['/thermo/Thermodynamics.h']
lines_to_patch = [33]

if __name__=='__main__':
    mpp_src = pathlib.Path('./mutationpp/src').resolve()

    for script, i_line in zip(scripts_to_patch, lines_to_patch):
        with open(str(mpp_src)+script, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for i, scriptline in enumerate(lines):
                if i==i_line-1: lines[i] = '#include <cassert>\n'

        with open(str(mpp_src)+script, 'w', encoding='utf-8') as f:
            f.writelines(lines)