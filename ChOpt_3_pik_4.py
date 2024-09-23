from glob import glob
import scipy.optimize as scop
import os, time, sys, shutil, subprocess
import numpy as np

pathway = "/home/odud/SG221_TiO6_orca_calc/embopt_Ti_large2"
timedecay, timelimit = 60, 130000 # sec
N_target_atoms, rowofreadinggrad = 44, 9

flag_to_active_string = "@"
start_time, round_n = time.time(), 8
startfile = glob(pathway + "/*.restart")[0] if len(glob(pathway + "/*.restart")) > 0 else glob(pathway + "/*.start")[0]
actfile, filename4tmp = startfile.split(".")[0], "/" + "ChOpt_tmp"
group_info, statlist, Charges8grads = {}, ["inqueue", "run", "faild", "done"], []
initfilename, logflag = (glob(pathway + "/*.logs")[0], True) if len(glob(pathway + "/*.logs")) > 0 else (glob(pathway + "/*.logf")[0], False)

class atom():
    def __init__(self, label, spatial_coord, charge=None, limits=None, group=None, sequence_number=None):
        self.label = label
        self.sc = spatial_coord
        self.q = charge
        self.lim = limits
        self.g = group
        self.n = sequence_number
    def go2str(self, group=False):
        spco_str, lims_str = "", str(min(self.lim)) + " " + str(max(self.lim))
        for value in self.sc:
            spco_str += str(value) + " "
        if self.label.lower() == "q":
            return self.label + " " + str(round(self.q, round_n)) + " " + spco_str + (flag_to_active_string if group else "#") + " " + self.g + " " + lims_str + "\n"
        else:
            return self.label + " " + spco_str + "Z = " + str(round(self.q, round_n)) + " " + (flag_to_active_string if group else "#") + " " + self.g + " " + lims_str + "\n"

class calc():
    def __init__(self, path, qcomp_shifted, status=statlist[0], noda=None):
        self.path = path
        self.stat = status
        self.qs = qcomp_shifted
        self.noda = noda
    def __eq__(self, other):
        if isinstance(other, calc):
            return self.path == other.path and sum(self.qs == other.qs) == len(self.qs)
        elif isinstance(other, list):
            return sum(self.qs == other) == len(self.qs) 
        else:
            return NotImplemented
    def launch(self):
        jobfile, outfile, intfile = self.path + filename4tmp + ".job", self.path + filename4tmp + ".out", self.path + filename4tmp + ".inp"
        assert os.system("ssh " + self.noda) == 0
        assert os.system("cd " + self.path) == 0
        vars_orca = ["module load mpi/ompi-4.1",
                    "export PATH=/home/odud/orca_5_0_4_linux_x86-64_shared_openmpi411/:$PATH",
                    "export LD_LIBRARY_PATH=/home/odud/orca_5_0_4_linux_x86-64_shared_openmpi411/:$LD_LIBRARY_PATH"]
        with open(jobfile, "w") as jobtxt:
            jobtxt.write("#!/bin/sh\n")
            for var_orca in vars_orca:
                jobtxt.write(var_orca + "\n")
            jobtxt.write("/home/odud/orca_5_0_4_linux_x86-64_shared_openmpi411/orca " + intfile + " > " + outfile + " &")
        with open(self.path + filename4tmp + ".nodes", "w") as nodestxt:
            nodestxt.write(self.noda + "\n")
        assert os.system("chmod +x " + jobfile) == 0
        assert os.system(jobfile) == 0
        assert os.system("exit") == 0
        return 0
    def creatdir(self):
        if os.path.exists(self.path):
            if self.stat in {statlist[0], statlist[-2], statlist[-1]}:
                shutil.rmtree(self.path)
                os.makedirs(self.path)
                return True
            else:
                return False
        else:
            os.makedirs(self.path)
            return True
    def removedir(self):
        shutil.rmtree(self.path)
        self.stat = statlist[0]
        return True

class ch8grad():
    def __init__(self, qcomp_shifted, resp_value):
        self.qs = qcomp_shifted
        self.v = resp_value
    def __eq__(self, other):
        if isinstance(other, ch8grad):
            return round(self.v, round_n) == round(other.v, round_n) and sum(self.qs == other.qs) == len(self.qs)
        elif isinstance(other, list) and len(self.qs) == len(other):
            return sum(self.qs == other) == len(self.qs)
        else:
            return NotImplemented

def listcompare(list1, list2):
    if len(list1) == len(list2):
        for i in range(len(list1)):
            if round(list1[i] - list2[i], round_n) != 0:
                return False
        return True
    else:
        return  False

def findid(qcomp_searched, ch8grad_set):
    for i, ch8grad_obj in enumerate(ch8grad_set):
        if listcompare(qcomp_searched, ch8grad_obj.qs):
            return i
    return -1

def reader_startfile():
    with open(startfile, "r") as starttxt:
        Charges = []
        for i, row in enumerate(starttxt):
            if flag_to_active_string in row:
                with open(actfile + ".inp", "a") as inptxt:
                    row_sep = row.split(flag_to_active_string)
                    inptxt.write(row_sep[0] + "\n")
                    row_sep_1, row_sep_2 = row_sep[0].split(), row_sep[1].split()
                    if row_sep_1[0].lower() == "q":
                        Charges.append(atom(row_sep_1[0], np.array(row_sep_1[2:5], dtype=float), float(row_sep_1[1]), np.array(row_sep_2[-2:], dtype = float), row_sep_2[-3], i))
                    else:
                        Charges.append(atom(row_sep_1[0], np.array(row_sep_1[1:4], dtype=float), float(row_sep_1[-1]), np.array(row_sep_2[-2:], dtype = float), row_sep_2[-3], i))                     
            else:
                with open(actfile + ".inp", "w" if i == 0 else "a") as inptxt:
                    inptxt.write(row)
    return Charges

def dashspliter(node, node_type):
    node_min, node_max = node.split("-")
    nodes_list_tmp = []
    for i in range(int(node_min), int(node_max)+1):
        node_tmp = node_type
        if i < 10:
            node_tmp = node_tmp + "00" + str(i)
        elif i < 100:
            node_tmp = node_tmp + "0" + str(i)
        else:
            node_tmp = node_tmp + str(i)
        nodes_list_tmp += [node_tmp]
    return nodes_list_tmp

def obtainer_nodesnames(initfilename):
    with open(initfilename, "r") as inputfile:
        jobid = inputfile.readline().split()[-1]
    assert subprocess.call("squeue -u odud > " + initfilename, shell=True, cwd=pathway+"/") == 0
    time.sleep(1)
    nodes_raw = None
    with open(initfilename, "r") as inputfile:
        for row in inputfile:
            if jobid in row:
                nodes_raw = row.strip().split()[-1]
                break
    assert subprocess.call("rm " + initfilename, shell=True, cwd=pathway+"/") == 0
    time.sleep(1)
    if not (nodes_raw is None):
        node_type = nodes_raw[0]
        flag1, flag2 = "," in nodes_raw, "-" in nodes_raw
        if flag1 or flag2:
            node_serias = nodes_raw.split("[")[-1].split("]")[0]
            active_nodes = []
            if flag1 and flag2:
                for node in node_serias.split(","):
                    active_nodes += dashspliter(node, node_type) if "-" in node else [node_type + node]
            elif flag1 and not flag2:
                active_nodes = [node_type + node for node in node_serias.split(",")]
            else:
                active_nodes = dashspliter(node_serias, node_type)
        else:
            active_nodes = [nodes_raw]
        with open(initfilename[:-1] + "f", "w") as outtxt:
            for node in active_nodes:
                outtxt.write(str(node) + "\n")
        return active_nodes
    else:
        raise FileNotFoundError("Sequense of nodes cannot be found")
    
def obtainer_nodesnames_restart(initfilename):
    with open(initfilename, "r") as inttxt:
        return [row.strip() for row in inttxt]

def periodicputter(nodes_list, n_qcomp):
    nodelist2qcompset = []
    n_nodes = len(nodes_list)
    j = 0
    for i in range(n_qcomp + 1):
        if j >= n_nodes:
            j -= n_nodes
        nodelist2qcompset.append(nodes_list[j])
        j += 1
    return nodelist2qcompset

def updater_intfile(Charges, path2actfile = actfile):
    with open(startfile, "r") as starttxt:
        with open(path2actfile + ".inp", "w") as inttxt:
            for i, row in enumerate(starttxt):
                if Charges[0].n <= i <= Charges[-1].n:
                    inttxt.write(Charges[i-Charges[0].n].go2str())
                else:
                    inttxt.write(row)
    return True

def updater_startfile(Charges):
    with open(actfile + ".inp", "r") as intfile:
        with open(actfile + ".restart", "w") as startfile:
            for i, row in enumerate(intfile):
                if Charges[0].n <= i <= Charges[-1].n:
                    startfile.write(Charges[i-Charges[0].n].go2str(True))
                else:
                    startfile.write(row)
    return True

def tracer_group(Charges):
    for i, atom_obj in enumerate(Charges):
        if atom_obj.g in group_info.keys():
            group_info[atom_obj.g] += [i]
        else:
            group_info[atom_obj.g] = [i]
    return True

def compression_of_charges(Charges):
    Charges_comp = []
    for key in group_info.keys():
        Charges_comp.append(Charges[group_info[key][0]].q)
    return Charges_comp

def giver_of_limits(Charges):
    left_border, right_border = [], []
    for key in group_info.keys():
        left_border.append(round(min(Charges[group_info[key][0]].lim), round_n))
        right_border.append(round(max(Charges[group_info[key][0]].lim), round_n))
    return list(zip(left_border, right_border))

def recover_of_charges(Charge_comp, Charge_old):
    for i_group, key in enumerate(group_info.keys()):
        for i in group_info[key]:
            Charge_old[i].q = Charge_comp[i_group]
    return Charge_old

def reader_outfile(path2actfile = actfile):
    flag_to_active_string = "CARTESIAN GRADIENT"
    with open(path2actfile + ".out", "r") as outtxt:
        RMS_tmp, flag = 0, False
        for i, row in enumerate(outtxt):
            if flag_to_active_string == row.strip():
                flag, countstart, countend = True, i + rowofreadinggrad, i + rowofreadinggrad + N_target_atoms
            elif flag:
                if countstart <= i <= countend:
                    atomic_grad = np.array(row.strip().split()[3:], dtype = float)
                    RMS_tmp += sum((atomic_grad)**2)
                elif i > countend:
                    flag = False
            else:
                continue
        return (RMS_tmp/N_target_atoms)**0.5

def updater_logfile(grad_value, charge_comp):
    with open(actfile + ".log", "a") as logtxt:
        logtxt.write( "time:\t" + str(round(time.time() - start_time, 1)) +" sec \tgrad:\t" + str(round(grad_value, round_n)) + " a.u.\n")
        for value in charge_comp:
            logtxt.write(str(value) + "\n")
    return True

qinit = reader_startfile()
assert tracer_group(qinit) # group_info updater
assert updater_intfile(qinit)

qcompinit, qcompinit_limits = compression_of_charges(qinit), giver_of_limits(qinit)
eps, ftol, nodes, maxit = [float(i) for i in sys.argv[1:]]
timelimit = timelimit * len(qcompinit) // nodes

nodelist_respond_qcompset = periodicputter(obtainer_nodesnames(initfilename) if logflag else obtainer_nodesnames_restart(initfilename), len(qcompinit))

def initialization_steps(qcomp):
    qcomp_set = [qcomp]
    for i in range(len(qcomp)):
        q_tmp = (qcomp[i] + eps) if ((qcomp[i] + eps) < max(qcompinit_limits[i])) else (qcomp[i] - eps)
        qcomp_set.append([(qcomp[j] if j != i else q_tmp) for j in range(len(qcomp))])
    return qcomp_set

def pre2start(qcomp_set):
    wait_list = []
    for i, qcomp_i in enumerate(qcomp_set):
        wait_list.append(calc(pathway + "/tmp/grad_" + str(i), qcomp_i, noda=nodelist_respond_qcompset[i]))
        assert wait_list[-1].creatdir()
        shutil.copy2(actfile + ".inp", wait_list[-1].path + filename4tmp + ".inp")
        assert updater_intfile(recover_of_charges(wait_list[-1].qs, qinit), wait_list[-1].path + filename4tmp)
    return wait_list

def periodiccall(act_set):
    successend, timeactual, searchlimit = "****ORCA TERMINATED NORMALLY****", time.time(), 10
    while time.time() - timeactual < timelimit:
        time.sleep(timedecay)
        for calc_i in act_set:
            if os.path.exists(calc_i.path + filename4tmp + ".out"):
                with open(calc_i.path + filename4tmp + ".out", "r") as outtxt:
                    for j, row in enumerate(reversed(list(outtxt))):
                        if j < searchlimit:
                            if successend in row:
                                calc_i.stat = statlist[-1]
                                break
                            else:
                                continue
                        else:
                            break
            else:
                calc_i.stat = statlist[-2]
        if sum([calc_i.stat == statlist[-1] for calc_i in act_set]) == len(act_set):
            return True
        else:
            continue
    return False

def start2calc(wait_list):
    while len(wait_list) > 0:
        run_list = []
        while len(wait_list) > 0 and len(run_list) < nodes:
            run_list.append(wait_list.pop(0))
            assert run_list[-1].launch() == 0
            run_list[-1].stat = statlist[1]
        if periodiccall(run_list):
            for calc_i in run_list:
                Charges8grads.append(ch8grad(calc_i.qs, reader_outfile(calc_i.path + filename4tmp)))
                calc_i.removedir()
        else:
            return False
    return True

def TF(qcomp):
    qcomp_id = findid(qcomp, Charges8grads)
    while qcomp_id == -1:
        qcomplist = initialization_steps(qcomp)
        waitlist = pre2start(qcomplist)
        if start2calc(waitlist):
            qcomp_id = findid(qcomp, Charges8grads)
            break
        else:
            timelimit += 7200
    updater_logfile(Charges8grads[qcomp_id].v, Charges8grads[qcomp_id].qs)
    return round(Charges8grads[qcomp_id].v, round_n)

totqcomp_init = sum(qcompinit)
cons = ({'type': 'eq', 'fun' : lambda qcomp: round(sum(qcomp) - totqcomp_init, round_n) })

res = scop.minimize(TF, qcompinit, args=(), method='SLSQP', jac=None, 
                    bounds=qcompinit_limits, constraints=cons, tol=None, callback=None, 
                    options={'disp': False, 'eps': eps, 'maxiter': maxit, 'ftol': ftol})
print(res)
assert updater_startfile(recover_of_charges(res.x, qinit))
