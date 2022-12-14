# encoding: utf-8
from .executor import LOG_FILE, write_log, COM_FILE, BOCA_FILE
import random, time, copy
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm

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
                 selection_strategy=['boca', 'local'][0],
                 budget=60, initial_sample_size=2):
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
        self.budget = budget  # n_iteration
        self.initial_sample_size = initial_sample_size

    def get_acc(self, model, steps):
        wait_for_predict = []
        pre_sp = []
        file = open("test.txt")
        while 1:
            lis = []
            line = file.readline().strip('\n')
            for i in range(len(line)):
                if i == 0:
                    continue
                elif i == len(line) - 1:
                    break
                else:
                    if line[i] == ',' or line[i] == ' ':
                        continue
                    else:
                        lis.append(int(line[i]))
            if not line:
                break
            if line == '\n':
                continue
            wait_for_predict.append(lis)
        file.close()
        t = 1
        estimators = model.estimators_
        for e in estimators:
            tmp = e.predict(np.array(wait_for_predict))
            if t == 1:
                for i in range(len(tmp)):
                    pre_sp.append(tmp[i])
                    t = t + 1
            else:
                for i in range(len(tmp)):
                    pre_sp[i] = pre_sp[i] + tmp[i]
                    t = t + 1
            for i in range(len(pre_sp)):
                pre_sp[i] = pre_sp[i] / (t - 1)
        acc_predict = []
        if steps != 1:
            true_running = [self.get_objective_score(indep, k_iter=0) for indep in wait_for_predict]
        else:
            true_running = []
        for i in range(len(pre_sp)):
            acc_predict.append(abs(pre_sp[i] - true_running[i]) / true_running)
        write_log(str(true_running), BOCA_FILE)
        write_log(str(pre_sp), BOCA_FILE)
        write_log(str(acc_predict), BOCA_FILE)
        write_log(str(np.mean(acc_predict)), BOCA_FILE)
        write_log('----------------', BOCA_FILE)
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

    def boca_search(self, model, eta, rnum, steps):
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

        # get rnum less-impactful sequences for each
        b2 = time.time()
        neighbors = []  # candidate seq
        # print('rnum: ' + str(rnum))
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

    def local_search(self, model, eta, incumbents):
        """
        Get 100 mutated + 10000 randomly sampled candidate optimization sequences,
        then compute Expected Improvement.

        :param incumbents: previous sequences, sorted by EI
        :return: 10100-size list of [EI, seq]
        """
        neighbors = []
        estimators = model.estimators_

        # TODO: get 100 candidate seq by flipping:
        #  1. seq in train_indep
        #  2. mutated seq in train_indep
        for inc0 in incumbents:
            num = 0
            inc = inc0  # best searching point
            while num < 100:
                neighbors_for_i = []
                for i in inc[0]:
                    tmp = copy.deepcopy(inc[0])
                    tmp[i] = 1 - tmp[i]  # filp
                    neighbors_for_i.append(tmp)
                pred_tmp = []
                for e in estimators:
                    pred_tmp.append(e.predict(np.array(neighbors_for_i)))
                acq_val_incumbent = self.get_ei(pred_tmp, eta)
                merged_neighbor_ei = zip(neighbors_for_i, acq_val_incumbent)
                sort_merged_neighbor_ei = sorted(merged_neighbor_ei, key=lambda x: x[1], reverse=True )
                max_n = sort_merged_neighbor_ei[0][0]
                max_ei = sort_merged_neighbor_ei[0][1]
                num += 1
                if max_ei <= inc[1]:
                    break
                inc = (max_n, max_ei)
            neighbors.append(inc[0])

        # TODO: get candidate seq by randomly sample
        for _ in range(10000):
            x = random.randint(0, 2 ** self.s_dim)
            x = self.generate_random_conf(x)
            neighbors.append(x)

        pred = []
        for e in estimators:
            pred.append(e.predict(np.array(neighbors)))
        acq_val_incumbent = self.get_ei(pred, eta)

        return [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]

    def get_training_sequence(self, training_indep, training_dep, eta, rnum, steps):
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(training_indep), np.array(training_dep))
        begin = time.time()
        if self.selection_strategy == 'local':
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
            merged_predicted_objectives = self.boca_search(model, eta, rnum, steps)
            # if steps == 1 or steps % 20 == 0 or steps == 58:
            #     self.get_acc(model,steps)
        merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
        end = time.time()
        print('search time: ' + str(end-begin))
        print('num: ' + str(len(merged_predicted_objectives)))

        # return unique seq in candidate set with highest EI
        begin = time.time()
        for x in merged_predicted_objectives:
            if x[0] not in training_indep:
                # true_speedup = self.get_objective_score(x[0])
                # if (abs(true_speedup - x[1]) / true_speedup) < 0.0001 :

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
        steps = 0
        merge = zip(training_indep, training_dep)
        merge_sort = [[indep, dep] for indep, dep in merge]
        merge_sort = sorted(merge_sort, key=lambda m: abs(m[1]), reverse=True)
        global_best_dep = merge_sort[0][1]  # best objective score
        global_best_indep = merge_sort[0][0]  # corresponding indep
        if self.decay:
            sigma = -self.scale ** 2 / (2 * math.log(self.decay))  # sigma = - scale^2 / 2*log(decay)
        else:
            sigma = None
        while self.initial_sample_size + steps < self.budget:
            steps += 1
            if self.decay:
                # C_i = C_0 * exp(-max(0,i-offset)^2 / (2*sigma^2))
                rnum = int(self.rnum0) * math.exp(-max(0, len(training_indep) - self.offset) ** 2 / (2*sigma**2))  # decay
            else:
                rnum = int(self.rnum0)
            rnum = int(rnum)
            # get best optimimzation sequence
            best_solution, _, num = self.get_training_sequence(training_indep, training_dep, global_best_dep, rnum, steps)
            ts.append(time.time()-begin)

            # add to training set, record time spent, score for this sequence
            training_indep.append(best_solution)
            best_result = self.get_objective_score(best_solution, k_iter=(self.initial_sample_size+steps))
            training_dep.append(best_result)
            if abs(best_result) > abs(global_best_dep):
                global_best_dep = best_result
                global_best_indep = best_solution

            ss = '{}: step {}, best {}, cur-best {}, independent-number {} , solution {}'.format(str(round(ts[-1])),
                                                                                     str(steps),
                                                                                     str(global_best_dep),
                                                                                     str(best_result),
                                                                                     str(num),
                                                                                     str(best_solution))
            write_log(ss, LOG_FILE)
        write_log(str(global_best_indep)+'\n=======================\n', LOG_FILE)
        return training_dep, ts