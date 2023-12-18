
import os,sys
import random, time, copy,subprocess
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm

all_flags = ['-falign-labels', '-fauto-inc-dec', '-fbranch-count-reg', '-fcaller-saves', '-fcode-hoisting', '-fcombine-stack-adjustments', '-fcompare-elim',
             '-fcprop-registers', '-fcrossjumping', '-fcse-follow-jumps', '-fdefer-pop', '-fdevirtualize', '-fdevirtualize-speculatively', '-fearly-inlining', 
             '-fexpensive-optimizations', '-fforward-propagate', '-ffunction-cse', '-fgcse', '-fguess-branch-probability', '-fhoist-adjacent-loads', '-fif-conversion', 
             '-fif-conversion2', '-findirect-inlining', '-finline-atomics', '-finline-functions-called-once', '-finline-small-functions', '-fipa-bit-cp', '-fipa-cp', 
             '-fipa-icf', '-fipa-icf-functions', '-fipa-icf-variables', '-fipa-profile', '-fipa-pure-const', '-fipa-ra', '-fipa-reference', '-fipa-sra', '-fipa-vrp', 
             '-fira-share-save-slots', '-fisolate-erroneous-paths-dereference', '-fjump-tables', '-flra-remat', '-fmove-loop-invariants', '-fomit-frame-pointer', 
             '-foptimize-sibling-calls', '-foptimize-strlen', '-fpartial-inlining', '-fpeephole2', '-fprefetch-loop-arrays', '-freg-struct-return', '-freorder-blocks', 
             '-freorder-blocks-and-partition', '-freorder-functions', '-frerun-cse-after-loop', '-fsched-critical-path-heuristic', '-fsched-group-heuristic', '-fsched-last-insn-heuristic',
             '-fsched-spec', '-fsched-stalled-insns-dep', '-fschedule-insns2', '-fshrink-wrap', '-fsigned-zeros', '-fsplit-wide-types', '-fssa-phiopt', '-fstore-merging', 
             '-fstrict-aliasing', '-fthread-jumps', '-ftrapping-math', '-ftree-bit-ccp', '-ftree-builtin-call-dce', '-ftree-ccp', '-ftree-ch', '-ftree-coalesce-vars', 
             '-ftree-copy-prop', '-ftree-dce', '-ftree-dominator-opts', '-ftree-dse', '-ftree-fre', '-ftree-loop-im', '-ftree-loop-optimize', '-ftree-pre', '-ftree-pta', 
             '-ftree-scev-cprop', '-ftree-sink', '-ftree-slsr', '-ftree-sra', '-ftree-switch-conversion', '-ftree-tail-merge', '-ftree-ter', '-ftree-vrp', '-fvar-tracking', '-fweb']


LOG_DIR = 'log' + os.sep
LOG_FILE = LOG_DIR + 'com_recordc1.log'
ERROR_FILE = LOG_DIR + 'err.log'

def write_log(ss, file):
    log = open(file, 'a')
    log.write(ss + '\n')
    log.flush()
    log.close()

def execute_terminal_command(command):
    """
    Execute the compiler and run command
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("命令执行成功！")
            if result.stdout:
                print("命令输出：")
                print(result.stdout)
        else:
            print("命令执行失败。")
            if result.stderr:
                print("错误输出：")
                print(result.stderr)
    except Exception as e:
        print("执行命令时出现错误：", str(e))

def get_objective_score(independent, k_iter):
    """
    Obtain the speedup
    """
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt = opt + all_flags[i] + ' '
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
            opt = opt + negated_flag_name + ' '
    print(opt)
    time_start = time.time()
    command = "gcc -O2 " + opt + " -c /home/zmx/BOCA_v2.0/benchmarks/cbench/automotive_bitcount/*.c"
    execute_terminal_command(command)
    command2 = "gcc -o a.out -O2 " + opt + " -lm *.o"
    execute_terminal_command(command2)
    command3 = "./a.out 1125000"
    execute_terminal_command(command3)
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)

    time_end = time.time()  
    time_c = time_end - time_start   #time opt

    time_o3 = time.time()
    command = "gcc -O3 -c /home/zmx/BOCA_v2.0/benchmarks/cbench/automotive_bitcount/*.c"
    execute_terminal_command(command)
    command2 = "gcc -o a.out -O3 -lm *.o"
    execute_terminal_command(command2)
    command3 = "./a.out 1125000"
    execute_terminal_command(command3)
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)

    time_o3_end = time.time()  
    time_o3_c = time_o3_end - time_o3   #time o3
    print(time_o3_c /time_c)

    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    return (time_o3_c /time_c)


opt_seq = [0, 1]
ts_tem = []


    
class compTuner:
    def __init__(self, dim, c1, c2, w, get_objective_score, random):
        """
        :param dim: number of compiler flags
        :param c1: parameter of pso process
        :param c2: parameter of pso process
        :param w: parameter of pso process
        :param get_objective_score: obtain true speedup
        :param random: random parameter
        """
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim
        self.V = []
        self.pbest = [] # best vector of each particle
        self.gbest = [] # best performance of each particle
        self.p_fit = [] # best vector of all particles
        self.fit = 0 # best performance of all particles
        self.get_objective_score = get_objective_score # function
        self.random = random

    def generate_random_conf(self, x):
        """
        Generation 0-1 mapping for disable-enable options
        """
        comb = bin(x).replace('0b', '')
        comb = '0' * (self.dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        :param preds:Sequences' speedup for EI
        :param eta:global best speedup
        :return:the EI for a sequence
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f
    
    def get_ei_predict(self, model, now_best, wait_for_train):
        """
        :param model:RandomForest Model
        :param now_best:global best speedup
        :param wait_for_train:Sequences Set
        :return:the Sequences' EI
        """
        preds = []
        estimators = model.estimators_
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]

    def runtime_predict(self, model, wait_for_train):
        """
        :param model:model:RandomForest Model
        :param wait_for_train:Sequences Set
        :return: the speedup of Sequences Set
        """
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        for tree in estimators:
            predictions = tree.predict(wait_for_train)
            sum_of_predictions += predictions
        a = []
        average_prediction = sum_of_predictions / len(estimators)
        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], average_prediction[i]]
            a.append(x)
        return a
    
    def getPrecision(self, model, seq):
        """
        :param model:
        :param seq:
        :return: The precision of a sequence and true speedup
        """
        true_running = self.get_objective_score(seq, k_iter=100086)
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        return abs(true_running - acc_predict) / true_running, true_running
    
    def selectByDistribution(self, merged_predicted_objectives):
        """
        Assign probabilities for different flag combinations
        """
        # sequences = [seq for seq, per in merged_predicted_objectives]
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        """
        :return: model, initial_indep, initial_dep
        """
        inital_indep = []
        # randomly sample initial training instances
        time_begin = time.time()
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        inital_dep = [self.get_objective_score(indep, k_iter=0) for indep in inital_indep]
        ts_tem.append(time.time() - time_begin)
        ss = '{}: best_seq {}, best_per {}'.format(str(round(ts_tem[-1])), str(max(inital_dep)), str(inital_indep[inital_dep.index(max(inital_dep))]))
        write_log(ss, LOG_FILE)
        all_acc = []
        model = RandomForestRegressor(random_state=self.random)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 2
        while rec_size <50:
            global_best = max(inital_dep)
            estimators = model.estimators_
            neighbors = []
            while len(neighbors) < 30000:
                x = random.randint(0, 2 ** self.dim)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)

            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    acc, lable = self.getPrecision(model, x[0])
                    inital_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1
            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[indx][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[indx][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                inital_dep.append(label)
                all_acc.append(acc)
                rec_size += 1
            ts_tem.append(time.time() - time_begin)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(ts_tem[-1])), str(max(inital_dep)), str(inital_indep[inital_dep.index(max(inital_dep))]))
            write_log(ss, LOG_FILE)
            model = RandomForestRegressor(random_state=self.random)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            if rec_size > 50 and np.mean(all_acc) < 0.04:
                break
        return model, inital_indep, inital_dep
    
    def getDistance(self, seq1, seq2):
        """
        :param seq1:
        :param seq2:
        :return: Getting the diversity of two sequences
        """
        t1 = np.array(seq1)
        t2 = np.array(seq2)
        s1_norm = np.linalg.norm(t1)
        s2_norm = np.linalg.norm(t2)
        cos = np.dot(t1, t2) / (s1_norm * s2_norm)
        return cos
    
    def init_v(self, n, d, V_max, V_min):
        """
        :param n: number of particles
        :param d: number of compiler flags
        :return: particle's initial velocity vector
        """
        v = []
        for i in range(n):
            vi = []
            for j in range(d):
                a = random.random() * (V_max - V_min) + V_min
                vi.append(a)
            v.append(vi)
        return v
    
    def update_v(self, v, x, m, n, pbest, g, w, c1, c2, vmax, vmin):
        """
        :param v: particle's velocity vector
        :param x: particle's position vector
        :param m: number of partical
        :param n: number of compiler flags
        :param pbest: each particle's best position vector
        :param g: all particles' best position vector
        :param w: weight parameter
        :param c1: control parameter
        :param c2: control parameter
        :param vmax: max V
        :param vmin: min V
        :return: each particle's new velocity vector
        """
        for i in range(m):
            a = random.random()
            b = random.random()
            for j in range(n):
                v[i][j] = w * v[i][j] + c1 * a * (pbest[i][j] - x[i][j]) + c2 * b * (g[j] - x[i][j])
                if v[i][j] < vmin:
                    v[i][j] = vmin
                if v[i][j] > vmax:
                    v[i][j] = vmax
        return v
    
    def run(self):
        """
        build model and get data set
        """
        ts = []
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        begin = time.time()
        self.V = self.init_v(len(inital_indep), len(inital_indep[0]), 10, -10)
        self.fit = 0
        self.pbest = list(inital_indep)
        self.p_fit = list(inital_dep)
        for i in range(len(inital_dep)):
            tmp = inital_dep[i]
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = inital_indep[i]
        end = time.time() + ts_tem[-1]
        ts.append(end - begin)
        ss = '{}: best {}, cur-best-seq {}'.format(str(round(end - begin)), str(self.fit), str(self.gbest))
        write_log(ss, LOG_FILE)
        t = 0
        while ts[-1] < 6000:
            if t == 0:
                self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest, self.gbest, self.w, self.c1, self.c2, 10, -10)
                for i in range(len(inital_indep)):
                    for j in range(len(inital_indep[0])):
                        a = random.random()
                        if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                            inital_indep[i][j] = 1
                        else:
                            inital_indep[i][j] = 0
                t = t + 1
            else:
                merged_predicted_objectives = self.runtime_predict(model, inital_indep)
                for i in range(len(merged_predicted_objectives)):
                    if merged_predicted_objectives[i][1] > self.p_fit[i]:
                        self.p_fit[i] = merged_predicted_objectives[i][1]
                        self.pbest[i] = merged_predicted_objectives[i][0]
                sort_merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
                current_best_seq = sort_merged_predicted_objectives[0][0]
                temp = self.get_objective_score(current_best_seq, 1000086)
                if  temp > self.fit:
                    self.gbest = current_best_seq
                    self.fit = temp
                    self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest,
                                           self.gbest, self.w, self.c1, self.c2, 10, -10)
                    for i in range(len(inital_indep)):
                        for j in range(len(inital_indep[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                                inital_indep[i][j] = 1
                            else:
                                inital_indep[i][j] = 0
                else:
                    """
                    Different update
                    """
                    avg_dis = 0.0
                    for i in range(1, len(merged_predicted_objectives)):
                        avg_dis = avg_dis + self.getDistance(merged_predicted_objectives[i][0], current_best_seq)
                    
                    avg_dis = avg_dis / (len(inital_indep) - 1)
                    
                    better_seed_indep = []
                    worse_seed_indep = []
                    better_seed_seq = []
                    worse_seed_seq = []
                    better_seed_pbest = []
                    worse_seed_pbest = []
                    better_seed_V = []
                    worse_seed_V = []
        
                    for i in range(0, len(merged_predicted_objectives)):
                        if self.getDistance(merged_predicted_objectives[i][0], current_best_seq) > avg_dis:
                            worse_seed_indep.append(i)
                            worse_seed_seq.append(merged_predicted_objectives[i][0])
                            worse_seed_pbest.append(self.pbest[i])
                            worse_seed_V.append(self.V[i])
                        else:
                            better_seed_indep.append(i)
                            better_seed_seq.append(merged_predicted_objectives[i][0])
                            better_seed_pbest.append(self.pbest[i])
                            better_seed_V.append(self.V[i])
                    """
                    update better particles
                    """
                    V_for_better = self.update_v(better_seed_V, better_seed_seq, len(better_seed_seq),
                                                 len(better_seed_seq[0]), better_seed_pbest, self.gbest
                                                 , self.w, 2 * self.c1, self.c2, 10, -10)
                    for i in range(len(better_seed_seq)):
                        for j in range(len(better_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_better[i][j])) > a:
                                better_seed_seq[i][j] = 1
                            else:
                                better_seed_seq[i][j] = 0
                    """
                    update worse particles
                    """
                    V_for_worse = self.update_v(worse_seed_V, worse_seed_seq, len(worse_seed_seq),
                                                len(worse_seed_seq[0]), worse_seed_pbest, self.gbest
                                                , self.w, self.c1, 2 * self.c2, 10, -10)
                    for i in range(len(worse_seed_seq)):
                        for j in range(len(worse_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_worse[i][j])) > a:
                                worse_seed_seq[i][j] = 1
                            else:
                                worse_seed_seq[i][j] = 0
                    for i in range(len(better_seed_seq)):
                        inital_indep[better_seed_indep[i]] = better_seed_seq[i]
                    for i in range(len(worse_seed_seq)):
                        inital_indep[worse_seed_indep[i]] = worse_seed_seq[i]
                t = t + 1

            ts.append(time.time() - begin + ts_tem[-1])
            ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(self.fit), str(self.gbest))
            write_log(ss, LOG_FILE)
            if (time.time() + ts_tem[-1] - begin) > 6000:
                break


if __name__ == "__main__":
    com_params = {}
    com_params['dim'] = len(all_flags)
    com_params['get_objective_score'] = get_objective_score
    com_params['c1'] = 2
    com_params['c2'] = 2
    com_params['w'] = 0.6
    com_params['random'] = 456


    com = compTuner(**com_params)
    dep, ts = com.run()