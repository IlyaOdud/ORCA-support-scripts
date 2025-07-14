import os, time, sys, shutil
from glob import glob
import re

init_exc = ("log_out", "out")
init_files = {}
sep_sign = "\\"
word_sign = "mult"

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

def write_for_python(table):
    tmp = re.sub(r"\),", "),\n", str(table))
    with open("out_for_python.txt", "w") as out_file:
        out_file.write(tmp)

def write_for_excel(table):
    with open("out_for_excel.txt", "w") as out_file:
        out_file.write("".join(f"{mult} {en}\n" for mult, en in table))

files = [choose_exc(name, init_exc) for name in filter(lambda word: word_sign in word.lower(), os.listdir())]

table = []
pattern1 = r"\*xyz *[0-9\-]+ *[0-9]+"
pattern2 = r"FINAL SINGLE POINT ENERGY.*\n"

table = [(obtain_info(file, pattern1).split()[-1], obtain_info(file, pattern2).split()[-1]) for file in files]

write_for_python(table)
write_for_excel(table)