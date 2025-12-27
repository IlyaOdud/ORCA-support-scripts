import os, time, sys, shutil
from glob import glob
import re

init_exc = ("log_out", "out")
init_files = {}
sep_sign = "/"
word_sign = "mult"
search_dirs = sys.argv[1:]
cwd_home = os.getcwd()


def choose_exc(name, excs):
    for exc in excs:
        tmp = glob(f"{name}{sep_sign}*.{exc}")
        if len(tmp) > 0:
            return tmp[-1]
    return None

def obtain_info(filename, pattern):
    with open(f"{filename}", "r") as file_link:
        total_file = file_link.read()
        list_obj = re.findall(pattern, total_file, flags=re.M)
        if len(list_obj) > 0:
            return list_obj[-1]
        else:
            raise FileExistsError(f"The pattern {pattern} wasn't find in {filename}")

def write_for_python(table, name):
    tmp = re.sub(r"\),", "),\n", str(table))
    with open(f"{name}.txt", "w") as out_file:
        out_file.write(tmp)




pattern1 = r"\*xyz *[0-9\-]+ *[0-9]+"
pattern2 = r"Final Gibbs free energy.*\n"

for search_dir in search_dirs:
    table = []
    
    os.chdir(f"{cwd_home}{sep_sign}{search_dir}")

    files = (
        choose_exc(name, init_exc)
        for name in filter(
            lambda word: word_sign in word.lower(),
            os.listdir()
            )
        )

    table = [
        (
            int(
                obtain_info(file, pattern1).split()[-1]
                ),
            float(
                obtain_info(file, pattern2).split()[-2]
                )
        )
        for file in files
        ]

    os.chdir(f"{cwd_home}")

    write_for_python(table, f"Gs_of_{search_dir}")
