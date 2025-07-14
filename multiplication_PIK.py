import os, time, sys, shutil
from glob import glob
import re

source_folder = "Source_files"
init_exc = ("inp", "job")
init_files = {}
sep_sign = "/"

def update_dict(init_files):
    if not os.path.exists(source_folder):
        raise FileExistsError(f"There isn't {source_folder}")
    else:
        for exc in init_exc:
            tmp = glob(f"{source_folder}{sep_sign}*.{exc}")
            if len(tmp) > 0:
                init_files.update({exc: tmp[0].split(sep_sign)[-1]})
            else:
                raise FileExistsError(f"There isn't .{exc} file")

update_dict(init_files)

def create_working_folders(list_of_mult):
    def creat_dir(mult):
        dirname = f"mult{mult}"
        while os.path.exists(dirname):
            dirname += "_sh"
        os.makedirs(dirname)
        return dirname
    actual_folders = []
    for mult in list_of_mult:
        tmp = creat_dir(mult)
        for file in init_files.values():
            shutil.copy2(f"{source_folder}{sep_sign}{file}", f"{tmp}")
        actual_folders.append(tmp)
    return actual_folders

def obtain_info(folder, filename, pattern):
    with open(f"{folder}{sep_sign}{filename}", "r") as file_link:
        total_file = file_link.read()
        match_obj = re.search(pattern, total_file, flags=re.M)
        if match_obj is not None:
            return match_obj.group(0)
        else:
            raise FileExistsError(f"The pattern {pattern} wasn't find in {filename}")

def write_info(folder, filename, pattern, word):
    path2file = f"{folder}{sep_sign}{filename}"
    with open(path2file, "r") as file_link:
        total_file = file_link.read()
    with open(path2file, "w") as file_link:
        total_file_mod = re.sub(pattern, word, total_file, flags=re.M)
        file_link.write(total_file_mod)

def modified_file(actual_folders, filename, list_of_mult, pattern):
    word_list = obtain_info(source_folder, init_files["inp"], pattern).split()
    for folder, mult in zip(actual_folders, list_of_mult):
        word_list[-1] = str(mult)
        word = " ".join(word_list)
        write_info(folder, filename, pattern, word)

list_of_mult = [*map(int, sys.argv[1:])]

actual_folders = create_working_folders(list_of_mult)
pattern = r"\*xyz *[0-9\-]+ *[0-9]+"
modified_file(actual_folders, init_files["inp"], list_of_mult, pattern)

cwd = os.getcwd()

def start_calc(cwd, folder, filename):
    os.chdir(f"{cwd}{sep_sign}{folder}{sep_sign}")
    time.sleep(2)
    os.system(f"sbatch {filename}")
    time.sleep(2)
    os.chdir(f"{cwd}{sep_sign}")

for folder in actual_folders:
    start_calc(cwd, folder, init_files["job"])
