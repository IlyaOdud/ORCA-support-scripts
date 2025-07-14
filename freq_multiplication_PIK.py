import os, shutil, sys, time
from glob import glob
import re

# constant variables ######################################################

source_folder = "freq_Source_files"
init_exc, init_exc_addition = ("inp", "job"), ("*out", "gbw")
sep_sign = "/"

############################################################################

def update_dict(source_folder, list_of_exc):
    init_files = {}
    if not os.path.exists(source_folder):
        raise FileExistsError(f"There isn't {source_folder}")
    else:
        for exc in list_of_exc:
            tmp = glob(f"{source_folder}{sep_sign}*.{exc}")
            if len(tmp) > 0:
                init_files.update({exc: tmp[0].split(sep_sign)[-1]})
            else:
                raise FileExistsError(f"There isn't .{exc} file")
    return init_files

init_files = update_dict(source_folder, init_exc)

def create_working_folders(list_of_mult, folder_name = "mult"):
    def creat_dir(mult):
        dirname = f"{folder_name}{mult}"
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

def get_folders(list_of_mults, folder_name):
    return [glob(f"{folder_name}{mult}")[0] for mult in list_of_mults]

def obtain_info(folder, filename, pattern, execute=lambda x: x[0]):
    with open(f"{folder}{sep_sign}{filename}", "r") as file_link:
        total_file = file_link.read()
        list_of_str = re.findall(pattern, total_file, flags=re.M)
        return execute(list_of_str)

def write_info(folder, filename, search_pattern, execute=lambda x:x):
    path2file = f"{folder}{sep_sign}{filename}"
    with open(path2file, "r") as file_link:
        total_file = file_link.read()
    with open(path2file, "w") as file_link:
        total_file_mod = re.sub(search_pattern, execute, total_file, flags=re.M)
        file_link.write(total_file_mod)

# function with parametrs ##################################################

def understand_active_atoms(list_of_str):
    def cut_trash(list_of_str):
        return [*map(lambda string: re.sub(r"[\{\}C]", "", string), list_of_str)]
    def extend_string(list_of_str):
        tmp = []
        for string in list_of_str:
            if ":" in string:
                n_min, n_max = tuple([*map(int, string.split(":"))])
                tmp.append([i for i in range(n_min, n_max + 1)])
            else:
                tmp.append(int(string))
        return tmp
    def unbox(list_of_lists_and_int):
        tmp = []
        for element in list_of_lists_and_int:
            if isinstance(element, list):
                tmp += element
            else:
                tmp += [element]
        return tmp
    def exclude(list_1, list_2):
        return [*filter(lambda x: x not in list_2, list_1)]
    
    list_of_constrained_atoms = unbox(extend_string(cut_trash(list_of_str)))
    n_min, n_max = 0, max(list_of_constrained_atoms)
    list_of_active_atoms = exclude([i for i in range(n_min, n_max + 1)], list_of_constrained_atoms)

    return list_of_active_atoms

def choose_active_atoms(raw_list_of_atoms, list_of_active_atoms):
    return [*map(lambda i: raw_list_of_atoms[i], list_of_active_atoms)]

def get_string(match_obj):
    return match_obj.group(0)

def define_string_replacer(list_of_active_atoms, active_atoms):
    counter, i, limit = 0, 0, len(active_atoms)
    def repalce_string(match_obj):
        nonlocal counter, i, limit
        string = get_string(match_obj)
        if i < limit and counter == list_of_active_atoms[i]:
            string = active_atoms[i]
            i += 1
        counter += 1
        return string
    return repalce_string

def define_pwd_replacer(folder, filename, code_pwd):
    def replace_pwd(match_obj):
        tmp = get_string(match_obj).split('"')
        return f'{tmp[0]}"{code_pwd}{sep_sign}{folder}{sep_sign}{filename}"'
    return replace_pwd

def define_mult_replacer(mult):
    def replace_mult(match_obj):
        tmp = get_string(match_obj).split()
        tmp = " ".join(tmp[:-1]) + " " +str(mult)
        return tmp
    return replace_mult

def read_file_old(filename, mode=-1):
    with open(filename, "r") as file_link:
        tmp_general = []
        i_start = None
        for i, string in enumerate(file_link):
            if i_start is None:
                if "CARTESIAN COORDINATES (ANGSTROEM)" in string:
                    i_start = i + 2
                    tmp_special = []
            else:
                if i >= i_start:
                    match_obj = re.search(r"([a-zA-Z>]{1,3})( +)([0-9.\-]+)( +[0-9.\-]+)?( +)([0-9.\-]+)( +)([0-9.\-]+)", string)
                    if match_obj is not None:
                        tmp_special += [match_obj.group(0)]
                    else:
                        i_start = None
                        tmp_general.append(tmp_special)
                else:
                    continue
        return tmp_general[mode]
############################################################################

op_lib = {
    "obtain_numbers": (r"[ \t]*{.*?}", understand_active_atoms),
    "write_all_atoms": (r"([a-zA-Z>]{1,3})( +)([0-9.\-]+)( +[0-9.\-]+)?( +)([0-9.\-]+)( +)([0-9.\-]+)", define_string_replacer),
    "write_path_of_gbw": (r'[ \t]*MoInp ".*?"', define_pwd_replacer),
    "write_mult": (r'[ \t]*\*xyz *[0-9\-]+ *[0-9]+', define_mult_replacer)
}

############################################################################

list_of_mults = [*map(int, sys.argv[1:])]

list_of_old_folders = get_folders(list_of_mults, "mult")
list_of_actual_folders = create_working_folders(list_of_mults, "freq_mult")
cwd = os.getcwd()

# operation with new folders ###############################################

for mult, old_folder, actual_folder in zip(list_of_mults, list_of_old_folders, list_of_actual_folders):

    folder_and_file = (actual_folder, init_files["inp"])

    list_of_active_atoms = obtain_info(*folder_and_file, op_lib["obtain_numbers"][0], op_lib["obtain_numbers"][1])
    actual_files = update_dict(old_folder, init_exc_addition)
    active_atoms = choose_active_atoms(read_file_old(f"{old_folder}{sep_sign}{actual_files['*out']}"), list_of_active_atoms)

    write_info(*folder_and_file, op_lib["write_all_atoms"][0], op_lib["write_all_atoms"][1](list_of_active_atoms, active_atoms))
    write_info(*folder_and_file, op_lib["write_path_of_gbw"][0], op_lib["write_path_of_gbw"][1](old_folder, actual_files["gbw"], cwd))
    write_info(*folder_and_file, op_lib["write_mult"][0], op_lib["write_mult"][1](mult))

    actual_files = {}

############################################################################

def start_calc(cwd, folder, filename):
    os.chdir(f"{cwd}{sep_sign}{folder}{sep_sign}")
    time.sleep(2)
    os.system(f"sbatch {filename}")
    time.sleep(2)
    os.chdir(f"{cwd}{sep_sign}")

for actual_folder in list_of_actual_folders:
    start_calc(cwd, actual_folder, init_files["job"])
