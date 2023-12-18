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

import os,sys,json
import random, time, copy,subprocess
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm

LOG_DIR = 'log' + os.sep
LOG_FILE = LOG_DIR + 'boca_recordc1.log'
ACC_FILE = LOG_DIR + 'acc.log'
time_end = 6000

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



class get_exchange(object):
    def __init__(self, incumbent):
        self.incumbent = incumbent  # fix values of impactful opts

    def to_next(self, opt_ids, l):
        """
        Flip selected less-impactful opt, then fix impactful optimization
        """
        ans = [0] * l
        for f in opt_ids:
            ans[f] = 1
        for f in self.incumbent:
            ans[f[0]] = f[1]
        return ans


class BOCA:
    def __init__(self, s_dim, get_objective_score, seed ,no_decay,
                 fnum=8, decay=0.5, scale=10, offset=20,
                 selection_strategy=['boca', 'local'][0], initial_sample_size=2):
        self.s_dim = s_dim
        self.get_objective_score = get_objective_score
        self.seed = seed

        self.fnum = fnum  # FNUM, number of impactful option
        if no_decay:
            self.decay = False
        else:
            self.decay = decay  # DECAY
            self.scale = scale  # SCALE
            self.offset = offset  # OFFSET
        self.rnum0 = 2**8  # base-number of less-impactful option-sequences, will decay

        self.selection_strategy = selection_strategy 
        self.initial_sample_size = initial_sample_size

    def generate_random_conf(self, x):
        """
        Generation 0-1 mapping for disable-enable options
        """

        comb = bin(x).replace('0b', '')
        comb = '0' * (self.s_dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        Compute Expected Improvements. (eta is global best indep)
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)
        # print('m:' + str(m))
        # print('s:' + str(s))

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m)*norm.cdf(z) + s * norm.pdf(z)
        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)

        return f

    def boca_search(self, model, eta, rnum):
        """
        Get 2**fnum * rnum candidate optimization sequences,
        then compute Expected Improvement.

        :return: 2**fnum  * rnum-size list of [EI, seq]
        """
        options = model.feature_importances_
        begin = time.time()
        opt_sort = [[i, x] for i, x in enumerate(options)]
        opt_selected = sorted(opt_sort, key=lambda x: x[1], reverse=True)[:self.fnum]
        opt_ids = [x[0] for x in opt_sort]
        neighborhood_iterators = []

        for i in range(2**self.fnum):  # search all combinations of impactful optimization
            comb = bin(i).replace('0b', '')
            comb = '0' * (self.fnum - len(comb)) + comb  # fnum-size 0-1 string
            inc = []  # list of tuple: (opt_k's idx, enable/disable)
            for k,s in enumerate(comb):
                if s == '1':
                    inc.append((opt_selected[k][0], 1))
                else:
                    inc.append((opt_selected[k][0], 0))
            neighborhood_iterators.append(get_exchange(inc))
        print('get impactful opt seq, using ' + str(time.time() - begin)+' s.')
        b2 = time.time()
        neighbors = []  # candidate seq
        for i, inc in enumerate(neighborhood_iterators):
            for _ in range(1 + rnum):
                flip_n = random.randint(0, self.s_dim)
                selected_opt_ids = random.sample(opt_ids, flip_n)
                neighbor_iter = neighborhood_iterators[i].to_next(selected_opt_ids, self.s_dim)
                neighbors.append(neighbor_iter)
        print('get '+str(len(neighbors))+' candidate seq, using '+str(time.time()-b2))

        preds = []
        estimators = model.estimators_
        b3 = time.time()
        for e in estimators:
            preds.append(e.predict(np.array(neighbors)))
        acq_val_incumbent = self.get_ei(preds, eta)
        print('get EI, using '+str(time.time() - b3)+' s.')

        return [[i,a] for a, i in zip(acq_val_incumbent, neighbors)]

    def get_training_sequence(self, training_indep, training_dep, eta, rnum):
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(training_indep), np.array(training_dep))

        # get candidate seqs and corresponding EI
        begin = time.time()
        if self.selection_strategy == 'local':
            # print('local search')
            estimators = model.estimators_
            preds = []
            for e in estimators:
                preds.append(e.predict(training_indep))
            train_ei = self.get_ei(preds, eta)
            configs_previous_runs = [(x, train_ei[i]) for i, x in enumerate(training_indep)]
            configs_previous_runs_sorted = sorted(configs_previous_runs, key=lambda x: x[1], reverse=True)[:10]
            merged_predicted_objectives = self.local_search(model, eta, configs_previous_runs_sorted)
        else:
            # print('boca search')
            merged_predicted_objectives = self.boca_search(model, eta, rnum)
        merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
        end = time.time()
        print('search time: ' + str(end-begin))
        print('num: ' + str(len(merged_predicted_objectives)))

        # return unique seq in candidate set with highest EI
        begin = time.time()
        for x in merged_predicted_objectives:
            if x[0] not in training_indep:
                print('get unique seq, using ' + str(time.time() - begin)+' s.')
                return x[0], x[1], len(merged_predicted_objectives)
    
   
    def run(self):
        """
        Run BOCA algorithm

        :return:
        """
        training_indep = []
        ts = []  # time spend
        begin = time.time()
        # randomly sample initial training instances
        while len(training_indep) < self.initial_sample_size:
            x = random.randint(0, 2**self.s_dim)
            initial_training_instance = self.generate_random_conf(x)
            # print(x, 2**self.s_dim,initial_training_instance)

            if initial_training_instance not in training_indep:
                training_indep.append(initial_training_instance)
                ts.append(time.time() - begin)

        training_dep = [self.get_objective_score(indep, k_iter=0) for indep in training_indep]
        write_log(str(training_dep), LOG_FILE)
        steps = 2
        merge = zip(training_indep, training_dep)
        merge_sort = [[indep, dep] for indep, dep in merge]
        merge_sort = sorted(merge_sort, key=lambda m: abs(m[1]), reverse=True)
        global_best_dep = merge_sort[0][1]  # best objective score
        global_best_indep = merge_sort[0][0]  # corresponding indep
        if self.decay:
            sigma = -self.scale ** 2 / (2 * math.log(self.decay))  # sigma = - scale^2 / 2*log(decay)
        else:
            sigma = None
        
        while ts[-1] < time_end:
            steps += 1
            if self.decay:
                rnum = int(self.rnum0) * math.exp(-max(0, len(training_indep) - self.offset) ** 2 / (2*sigma**2))  # decay
            else:
                rnum = int(self.rnum0)
            rnum = int(rnum)
            # get best optimimzation sequence
            best_solution, _, num = self.get_training_sequence(training_indep, training_dep, global_best_dep, rnum)
            ts.append(time.time()-begin)
            
            # add to training set, record time spent, score for this sequence
            training_indep.append(best_solution)
            best_result = self.get_objective_score(best_solution, k_iter=(self.initial_sample_size+steps))
            training_dep.append(best_result)
            if abs(best_result) > abs(global_best_dep):
                global_best_dep = best_result
                global_best_indep = best_solution
            
            ss = '{}: best {}, cur-best {}, independent-number {} , solution {}'.format(str(round(ts[-1])),
                                                                                    str(global_best_dep),
                                                                                    str(best_result),
                                                                                    str(num),
                                                                                    str(best_solution))
            print(ss)
            write_log(ss, LOG_FILE)
        write_log(str(global_best_indep)+'\n=======================\n', LOG_FILE)
        print(str(global_best_indep))
        return training_dep, ts


if __name__ == "__main__":

    boca_params = {}
    boca_params['s_dim'] = len(all_flags)
    boca_params['get_objective_score'] = get_objective_score
    boca_params['fnum'] = 8
    boca_params['decay'] = 0.5
    boca_params['no_decay'] = False
    boca_params['scale'] = 10
    boca_params['offset'] = 20
    boca_params['selection_strategy'] = 'boca'
    boca_params['initial_sample_size'] = 2
    boca_params['seed'] = 456


    boca = BOCA(**boca_params)
    dep, ts = boca.run()